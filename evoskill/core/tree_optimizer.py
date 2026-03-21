"""Tree-aware Prompt Optimizer with automatic splitting, pruning, and section-wise optimization.

This module provides advanced tree management capabilities:
1. Automatic splitting when detecting contradictory feedback
2. Automatic pruning based on performance metrics
3. Section-wise optimization (instruction/examples/constraints)
4. Bottom-up tree optimization
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from evoskill.core.abc import ModelAdapter, OptimizablePrompt, Experience, TextualGradient
from evoskill.core.optimizer import TrainFreeOptimizer
from evoskill.core.optimizer_config import OptimizerConfig, Validator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TreeOptimizerConfig:
    """Configuration for TreeAwareOptimizer.

    Parameters
    ----------
    auto_split : bool
        Automatically split skills when detecting contradictory feedback.
    auto_prune : bool
        Automatically prune low-performing child skills.
    prune_threshold : float
        Performance threshold for pruning (0.0-1.0).
    prune_protection_rounds : int
        Number of rounds to protect newly created nodes from pruning.
        This implements "progressive disclosure" - nodes need time to accumulate experience.
    prune_usage_threshold : int
        Minimum usage count before considering prune. New nodes start at 0.
    prune_strategy : str
        Pruning strategy: 'aggressive', 'moderate', 'conservative', 'disabled'.
        - aggressive: prune quickly (default behavior)
        - moderate: wait for more data
        - conservative: only prune clearly failing nodes
        - disabled: never auto-prune
    collapse_instead_of_prune : bool
        If True, collapse underused nodes (hide from routing) instead of deleting.
        This preserves learned knowledge for potential future use.
        Implements "progressive disclosure" of context.
    min_samples_for_split : int
        Minimum number of samples required before considering a split.
    max_tree_depth : int
        Maximum depth of the skill tree.
    optimization_order : str
        Tree traversal order: 'bottom_up' or 'top_down'.
    section : Optional[str]
        Which part of the prompt to optimize: 'all', 'instruction', 'examples', 'constraints'.
    """

    auto_split: bool = True
    auto_prune: bool = True
    prune_threshold: float = 0.3
    prune_protection_rounds: int = 2  # Protect new nodes for 2 rounds
    prune_usage_threshold: int = 2  # Minimum usage before considering prune
    prune_strategy: str = "moderate"  # aggressive/moderate/conservative/disabled
    collapse_instead_of_prune: bool = True  # Progressive disclosure
    min_samples_for_split: int = 5
    max_tree_depth: int = 3
    optimization_order: str = "bottom_up"  # bottom_up / top_down
    section: Optional[str] = "all"  # all / instruction / examples / constraints


# ---------------------------------------------------------------------------
# Result Classes
# ---------------------------------------------------------------------------

@dataclass
class TreeOptimizationResult:
    """Result of tree optimization.

    Attributes
    ----------
    tree : SkillTree
        The optimized skill tree.
    nodes_optimized : int
        Number of nodes optimized.
    splits_performed : int
        Number of split operations performed.
    prunes_performed : int
        Number of prune operations performed.
    total_steps : int
        Total optimization steps across all nodes.
    node_results : Dict[str, Any]
        Detailed results for each node (path -> result).
    """

    tree: Any  # SkillTree (avoid circular import)
    nodes_optimized: int = 0
    splits_performed: int = 0
    prunes_performed: int = 0
    total_steps: int = 0
    node_results: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TreeAwareOptimizer
# ---------------------------------------------------------------------------

class TreeAwareOptimizer:
    """Tree-aware optimizer with automatic splitting and pruning.

    This optimizer combines:
    1. Single-point optimization (via TrainFreeOptimizer)
    2. Tree management (splitting, pruning, traversal)

    Parameters
    ----------
    adapter : ModelAdapter
        The model adapter for generation and gradient computation.
    base_optimizer : Optional[TrainFreeOptimizer]
        The base optimizer for single-point optimization.
        If None, creates a default one.
    config : TreeOptimizerConfig
        Tree optimization configuration.
    base_optimizer_config : Optional[OptimizerConfig]
        Configuration for the base optimizer.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        base_optimizer: Optional[TrainFreeOptimizer] = None,
        config: Optional[TreeOptimizerConfig] = None,
        base_optimizer_config: Optional[OptimizerConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or TreeOptimizerConfig()

        # Create base optimizer if not provided
        if base_optimizer is None:
            base_config = base_optimizer_config or OptimizerConfig()
            self.base_optimizer = TrainFreeOptimizer(
                adapter=adapter,
                config=base_config,
            )
        else:
            self.base_optimizer = base_optimizer

    # ------------------------------------------------------------------
    # Main Optimization Methods
    # ------------------------------------------------------------------

    def optimize_tree(
        self,
        tree: Any,  # SkillTree
        experiences: List[Experience],
        validator: Optional[Validator] = None,
    ) -> TreeOptimizationResult:
        """Recursively optimize every skill in a tree.

        Strategy: optimize leaves first, then parents (bottom-up).
        If auto_split is True, analyze each node for potential splits.
        If auto_prune is True, remove low-performing nodes.

        Parameters
        ----------
        tree : SkillTree
            The skill tree to optimize.
        experiences : List[Experience]
            List of experiences with feedback.
        validator : Optional[Validator]
            Optional validator for prompt evaluation.

        Returns
        -------
        TreeOptimizationResult
            The optimization result with metrics.
        """
        logger.info(f"Starting tree optimization with {len(experiences)} experiences")

        # Increment age for all nodes (progressive disclosure)
        def _increment_age(node):
            node.age += 1
            for child in node.children.values():
                _increment_age(child)

        _increment_age(tree.root)
        logger.info("Incremented age for all nodes")

        result = TreeOptimizationResult(tree=tree)

        # Traverse tree in specified order
        nodes = self._walk_tree(tree, order=self.config.optimization_order)

        for node_path, node in nodes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing node: {node_path}")
            logger.info(f"{'='*60}")

            # Step 1: Single-point optimization
            optimized_prompt = self._optimize_node(
                node=node,
                experiences=experiences,
                validator=validator,
                section=self.config.section,
            )

            if optimized_prompt:
                # Update node's prompt - Create new Skill object with updated prompt
                # ⭐ Pydantic models are immutable, so we need to create a new object
                new_prompt_text = self._extract_prompt_text(optimized_prompt)

                # DEBUG: Log the update
                logger.info(f"🔧 Updating node '{node.name}':")
                logger.info(f"   Old prompt (v{node.skill.version}): {node.skill.system_prompt[:100]}...")
                logger.info(f"   New prompt (v{optimized_prompt.version}): {new_prompt_text[:100]}...")

                # Use model_copy to create updated skill
                updated_skill = node.skill.model_copy(update={
                    'system_prompt': new_prompt_text,
                    'version': optimized_prompt.version,
                })

                # Verify updated_skill has new values
                logger.info(f"   ✅ Created updated_skill with prompt: {updated_skill.system_prompt[:100]}...")
                logger.info(f"   ✅ Created updated_skill with version: {updated_skill.version}")

                # Replace the node's skill with the updated one
                node.skill = updated_skill

                # Verify assignment worked
                logger.info(f"   ✅ After assignment, node.skill.prompt: {node.skill.system_prompt[:100]}...")
                logger.info(f"   ✅ After assignment, node.skill.version: {node.skill.version}")

                result.nodes_optimized += 1

            # Step 2: Auto-split analysis
            # 🚨 CRITICAL: Check depth limit before splitting
            # Calculate depth from node_path (number of dots + 1)
            current_depth = node_path.count('.') + 1 if node_path else 0

            if self.config.auto_split:
                # Check if we've reached max depth
                if current_depth >= self.config.max_tree_depth:
                    logger.info(
                        f"🚫 Node '{node.name}' at depth {current_depth} "
                        f"reached max_tree_depth={self.config.max_tree_depth}, skipping split"
                    )
                else:
                    specs = self.analyze_split_need(
                        prompt=optimized_prompt or self._get_node_prompt(node),
                        experiences=experiences,
                    )

                    if specs:
                        children = self.generate_child_prompts(
                            parent_prompt=optimized_prompt or self._get_node_prompt(node),
                            children_specs=specs,
                        )

                        # Add children to tree
                        for child_spec, child_prompt in zip(specs, children):
                            try:
                                tree.add_child(
                                    parent_path=node_path,
                                    child_name=child_spec["name"],
                                    skill=self._create_skill_from_prompt(child_prompt, node.skill),
                                    description=child_spec.get("description"),
                                )
                                result.splits_performed += 1
                            except Exception as e:
                                logger.error(f"Failed to add child '{child_spec['name']}': {e}")

            # Step 3: Auto-prune analysis (for children)
            if self.config.auto_prune and node.children:
                for child_name, child_node in list(node.children.items()):
                    metrics = self._collect_node_metrics(child_node, experiences)

                    if self.analyze_prune_need(child_node, metrics):
                        try:
                            tree.prune(f"{node_path}.{child_name}" if node_path else child_name)
                            result.prunes_performed += 1
                        except Exception as e:
                            logger.error(f"Failed to prune '{child_name}': {e}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Tree Optimization Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Nodes optimized: {result.nodes_optimized}")
        logger.info(f"Splits performed: {result.splits_performed}")
        logger.info(f"Prunes performed: {result.prunes_performed}")

        return result

    # ------------------------------------------------------------------
    # Helper Methods (to be implemented in subtasks)
    # ------------------------------------------------------------------

    def analyze_split_need(
        self,
        prompt: OptimizablePrompt,
        experiences: List[Experience],
    ) -> Optional[List[Dict]]:
        """Ask the LLM whether the prompt should be split into sub-prompts.

        Returns a list of child specs [{name, description, focus}]
        if a split is recommended, or None if not.

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to analyze.
        experiences : List[Experience]
            Experiences with feedback.

        Returns
        -------
        Optional[List[Dict]]
            List of child specs or None.
        """
        # Filter experiences with feedback
        diagnosed = [exp for exp in experiences if exp.get_feedback() is not None]

        # Check minimum samples
        if len(diagnosed) < self.config.min_samples_for_split:
            logger.debug(
                f"Not enough samples for split analysis: {len(diagnosed)} < {self.config.min_samples_for_split}"
            )
            return None

        # Build feedback block
        feedback_items = []
        for exp in diagnosed:
            user_input = exp.get_input()
            if isinstance(user_input, list):
                # Conversation format
                user_text = user_input[-1].get("content", "") if user_input else ""
            elif isinstance(user_input, dict):
                user_text = user_input.get("text", str(user_input))
            else:
                user_text = str(user_input)

            feedback = exp.get_feedback()
            if feedback:
                if hasattr(feedback, 'critique') and feedback.critique:
                    critique_text = feedback.critique
                elif hasattr(feedback, 'correction') and feedback.correction:
                    critique_text = f"Ideal: {feedback.correction}"
                else:
                    score = feedback.to_score() if hasattr(feedback, 'to_score') else 0.5
                    critique_text = f"Score: {score:.2f}"
            else:
                critique_text = "No feedback"

            feedback_items.append(
                f"- Task: \"{user_text[:100]}\"  Critique: \"{critique_text[:100]}\""
            )

        feedback_block = "\n".join(feedback_items)

        # Extract prompt text
        prompt_text = self._extract_prompt_text(prompt)

        # Build judge messages
        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are a prompt architecture expert. Analyze the feedback "
                    "below and decide whether this single System Prompt should "
                    "be SPLIT into specialised child prompts for different "
                    "domains/task-types.\n\n"
                    "**IMPORTANT: Splitting increases complexity and should ONLY be done when clearly necessary.**\n\n"
                    "SPLIT ONLY if you see CLEAR evidence of:\n"
                    "- **Strong contradictory requirements** (e.g., formal vs casual tone, detailed vs concise)\n"
                    "- **Fundamentally different task types** that cannot be handled by one prompt\n"
                    "- **Irreconcilable feedback conflicts** that require specialized approaches\n\n"
                    "DO NOT split if:\n"
                    "- The prompt can be improved with better wording\n"
                    "- Issues can be fixed with examples or clarifications\n"
                    "- The feedback suggests incremental improvements, not structural changes\n\n"
                    "If YES, return a JSON array of child specs (2-4 children max):\n"
                    '[{"name": "...", "description": "...", "focus": "..."}]\n\n'
                    "Each child should have:\n"
                    "- name: Short identifier (snake_case)\n"
                    "- description: Brief description of this specialization\n"
                    "- focus: What aspect this child focuses on\n\n"
                    "If NO (keep as single prompt), return exactly: null\n\n"
                    "**Default to NOT splitting unless there is overwhelming evidence.**\n\n"
                    "Return ONLY the JSON (or null). No commentary, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current System Prompt:\n\"\"\"\n{prompt_text}\n\"\"\"\n\n"
                    f"Feedback from different tasks:\n{feedback_block}\n\n"
                    "Should this prompt be split? If yes, return child specs."
                ),
            },
        ]

        # Call LLM
        try:
            raw = self.adapter._call_api(
                messages=judge_messages,
                system=None,
                temperature=0.3,
            )
            raw = raw.strip()

            # Parse response
            if raw.lower() == "null" or not raw.startswith("["):
                logger.info("LLM decided NOT to split")
                return None

            specs = json.loads(raw)

            if isinstance(specs, list) and len(specs) >= 2:
                # Validate each spec
                for spec in specs:
                    if not isinstance(spec, dict):
                        logger.warning("Invalid spec format (not a dict)")
                        return None
                    if "name" not in spec or "description" not in spec:
                        logger.warning("Spec missing required fields (name/description)")
                        return None
                    # Ensure 'focus' field exists
                    spec.setdefault("focus", spec.get("description", ""))

                logger.info(f"LLM recommends splitting into {len(specs)} children")
                return specs
            else:
                logger.warning(f"Invalid split response (expected list with >= 2 items): {specs}")
                return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse split analysis JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during split analysis: {e}", exc_info=True)
            return None

    def generate_child_prompts(
        self,
        parent_prompt: OptimizablePrompt,
        children_specs: List[Dict],
    ) -> List[OptimizablePrompt]:
        """Generate specialized prompts for each child spec.

        Parameters
        ----------
        parent_prompt : OptimizablePrompt
            The parent prompt.
        children_specs : List[Dict]
            List of child specs with name, description, focus.

        Returns
        -------
        List[OptimizablePrompt]
            List of child prompts.
        """
        from evoskill.core.prompts import TextPrompt

        # Extract parent prompt text
        parent_text = self._extract_prompt_text(parent_prompt)

        # Format specs for LLM
        specs_json = json.dumps(children_specs, ensure_ascii=False, indent=2)

        # Build rewrite prompt
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer. Given a parent System "
                    "Prompt and a list of child specialization specs, write a "
                    "tailored System Prompt for each child.\n\n"
                    "Guidelines:\n"
                    "- Each child prompt should inherit core structure from parent\n"
                    "- Specialize for the child's focus area\n"
                    "- Keep prompts concise and actionable\n"
                    "- Maintain consistency with parent's style\n\n"
                    "Return a JSON array where each item has:\n"
                    '- "name": same as input spec\n'
                    '- "description": same as input spec\n'
                    '- "system_prompt": the specialized prompt text\n\n'
                    "Return ONLY valid JSON. No commentary, no markdown code fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Parent System Prompt:\n\"\"\"\n{parent_text}\n\"\"\"\n\n"
                    f"Child specs:\n{specs_json}\n\n"
                    "Generate specialized prompts for each child."
                ),
            },
        ]

        try:
            # Call LLM
            raw = self.adapter._call_api(
                messages=rewrite_messages,
                system=None,
                temperature=0.5,
            )
            raw = raw.strip()

            # Clean up potential markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            # Parse response
            result = json.loads(raw)

            if not isinstance(result, list):
                logger.warning(f"Expected list, got {type(result)}")
                # Fallback: use parent prompt for all children
                result = [
                    {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "system_prompt": parent_text,
                    }
                    for spec in children_specs
                ]

            # Convert to OptimizablePrompt instances
            child_prompts = []
            for item in result:
                if "system_prompt" not in item:
                    logger.warning(f"Missing 'system_prompt' in result for {item.get('name', 'unknown')}")
                    item["system_prompt"] = parent_text

                # Create TextPrompt
                child_prompt = TextPrompt(content=item["system_prompt"])
                child_prompts.append(child_prompt)

            logger.info(f"Generated {len(child_prompts)} child prompts")
            return child_prompts

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse child prompts JSON: {e}")
            # Fallback: return parent prompt for all children
            from evoskill.core.prompts import TextPrompt
            return [TextPrompt(content=parent_text) for _ in children_specs]

        except Exception as e:
            logger.error(f"Error generating child prompts: {e}", exc_info=True)
            # Fallback
            from evoskill.core.prompts import TextPrompt
            return [TextPrompt(content=parent_text) for _ in children_specs]

    def analyze_prune_need(self, node: Any, metrics: Dict) -> bool:
        """Analyze whether a node should be pruned.

        Pruning conditions (strategy-dependent):
        1. Node performance < prune_threshold
        2. Usage frequency is extremely low (after protection period)
        3. Parent already covers its responsibilities

        Progressive Disclosure:
        - New nodes are protected for `prune_protection_rounds` rounds
        - Strategy controls how aggressive pruning is
        - Can collapse (hide) instead of delete if `collapse_instead_of_prune=True`

        Parameters
        ----------
        node : SkillNode
            The node to analyze.
        metrics : Dict
            Performance metrics for the node.

        Returns
        -------
        bool
            True if the node should be pruned.
        """
        # Get node age (rounds since creation)
        node_age = getattr(node, 'age', 0)  # Default to 0 if not tracked

        # Protection period for new nodes
        if node_age < self.config.prune_protection_rounds:
            logger.info(
                f"🛡️ Protecting '{node.name}': age {node_age} < protection period {self.config.prune_protection_rounds}"
            )
            return False

        # Get metrics
        performance_score = metrics.get("performance_score", 0.5)
        usage_count = metrics.get("usage_count", 0)
        success_rate = metrics.get("success_rate", 0.5)

        # Strategy-specific thresholds
        if self.config.prune_strategy == "disabled":
            return False

        elif self.config.prune_strategy == "conservative":
            # Only prune clearly failing nodes
            usage_threshold = max(self.config.prune_usage_threshold, 5)
            performance_threshold = 0.2
            success_threshold = 0.2

        elif self.config.prune_strategy == "moderate":
            # Balanced approach
            usage_threshold = self.config.prune_usage_threshold
            performance_threshold = self.config.prune_threshold
            success_threshold = 0.3

        elif self.config.prune_strategy == "aggressive":
            # Quick pruning
            usage_threshold = max(1, self.config.prune_usage_threshold - 1)
            performance_threshold = self.config.prune_threshold
            success_threshold = 0.4

        else:
            # Default to moderate
            usage_threshold = self.config.prune_usage_threshold
            performance_threshold = self.config.prune_threshold
            success_threshold = 0.3

        # Condition 1: Low performance score
        if performance_score < performance_threshold:
            action = "collapse" if self.config.collapse_instead_of_prune else "prune"
            logger.info(
                f"✂️ {action.capitalize()} '{node.name}': performance {performance_score:.2f} < threshold {performance_threshold}"
            )
            return True

        # Condition 2: Very low usage frequency (after protection period)
        if usage_count < usage_threshold:
            action = "collapse" if self.config.collapse_instead_of_prune else "prune"
            logger.info(
                f"✂️ {action.capitalize()} '{node.name}': low usage count ({usage_count} < {usage_threshold})"
            )
            return True

        # Condition 3: Low success rate
        if success_rate < success_threshold:
            action = "collapse" if self.config.collapse_instead_of_prune else "prune"
            logger.info(
                f"✂️ {action.capitalize()} '{node.name}': low success rate {success_rate:.2%} < {success_threshold:.2%}"
            )
            return True

        return False

    def _collect_node_metrics(self, node: Any, experiences: List[Experience]) -> Dict:
        """Collect performance metrics for a node.

        Returns metrics:
        - performance_score: Average feedback score
        - usage_count: Number of times used
        - success_rate: Percentage of positive feedback

        Parameters
        ----------
        node : SkillNode
            The node to collect metrics for.
        experiences : List[Experience]
            All experiences.

        Returns
        -------
        Dict
            Performance metrics.
        """
        # Filter experiences related to this node
        # In a real implementation, you'd need to track which node handled which experience
        # For now, use a simple heuristic: check if node name appears in metadata

        node_name = node.name.lower()
        relevant_experiences = []

        for exp in experiences:
            # Check if this experience might be related to this node
            # This is a simple heuristic - in production you'd have better tracking
            metadata = exp.metadata if hasattr(exp, 'metadata') else {}
            if metadata and isinstance(metadata, dict):
                skill_name = metadata.get("skill_name", "").lower()
                if skill_name and (node_name in skill_name or skill_name in node_name):
                    relevant_experiences.append(exp)

        # If no relevant experiences found, use node's usage_count attribute
        if not relevant_experiences:
            # 🔧 CRITICAL FIX: Don't use all experiences as fallback
            # Instead, use the node's tracked usage_count
            usage_count = getattr(node, 'usage_count', 0)
            return {
                "performance_score": 0.5,  # Neutral score for unknown performance
                "usage_count": usage_count,  # Use tracked usage count
                "success_rate": 0.5,  # Neutral success rate
            }

        # Calculate metrics
        if not relevant_experiences:
            return {
                "performance_score": 0.5,
                "usage_count": 0,
                "success_rate": 0.5,
            }

        scores = []
        positive_count = 0

        for exp in relevant_experiences:
            feedback = exp.get_feedback()
            if feedback:
                # Get score
                if hasattr(feedback, 'to_score'):
                    score = feedback.to_score()
                else:
                    # Estimate score from feedback type
                    score = 0.5  # neutral

                scores.append(score)
                if score >= 0.6:  # consider 0.6+ as positive
                    positive_count += 1

        # Calculate average metrics
        if scores:
            performance_score = sum(scores) / len(scores)
            success_rate = positive_count / len(scores)
        else:
            performance_score = 0.5
            success_rate = 0.5

        usage_count = len(relevant_experiences)

        return {
            "performance_score": performance_score,
            "usage_count": usage_count,
            "success_rate": success_rate,
        }

    def optimize_prompt_section(
        self,
        prompt: OptimizablePrompt,
        experiences: List[Experience],
        section: str = "all",
    ) -> OptimizablePrompt:
        """Optimize only a specific section of the prompt.

        Sections:
        - 'all': Complete optimization (default)
        - 'instruction': Only optimize instruction part
        - 'examples': Only optimize few-shot examples
        - 'constraints': Only optimize constraint conditions

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to optimize.
        experiences : List[Experience]
            Experiences to learn from.
        section : str
            Which section to optimize.

        Returns
        -------
        OptimizablePrompt
            Optimized prompt.
        """
        if section == "all":
            # Use regular optimization
            result = self.base_optimizer.optimize(
                prompt=prompt,
                experiences=experiences,
                validator=None,
            )
            return result.optimized_prompt

        # For section-wise optimization, we need to:
        # 1. Parse prompt into sections
        parts = self._parse_prompt_sections(prompt)

        # 2. Build section-specific rewrite prompt
        if section == "instruction":
            rewrite_prompt_content = self._build_section_rewrite_prompt(
                current_content=parts.get("instruction", ""),
                section="instruction",
                experiences=experiences,
            )
        elif section == "examples":
            rewrite_prompt_content = self._build_section_rewrite_prompt(
                current_content=parts.get("examples", ""),
                section="examples",
                experiences=experiences,
            )
        elif section == "constraints":
            rewrite_prompt_content = self._build_section_rewrite_prompt(
                current_content=parts.get("constraints", ""),
                section="constraints",
                experiences=experiences,
            )
        else:
            logger.warning(f"Unknown section: {section}, using full optimization")
            result = self.base_optimizer.optimize(
                prompt=prompt,
                experiences=experiences,
                validator=None,
            )
            return result.optimized_prompt

        # 3. Reconstruct prompt with updated section
        new_parts = parts.copy()
        new_parts[section] = rewrite_prompt_content

        new_content = self._assemble_prompt_sections(new_parts)

        # 4. Create new prompt with same metadata
        new_prompt = prompt.bump_version()
        if hasattr(new_prompt, 'content'):
            new_prompt.content = new_content
        elif hasattr(new_prompt, 'text'):
            new_prompt.text = new_content

        return new_prompt

    def _parse_prompt_sections(self, prompt: OptimizablePrompt) -> Dict[str, str]:
        """Parse prompt into structural sections.

        This is a simple parser that looks for:
        - Clear headers (INSTRUCTION:, EXAMPLE:, CONSTRAINT:)
        - Bullet points
        - Numbered lists

        Parameters
        ----------
        prompt : OptimizablePrompt
            The prompt to parse.

        Returns
        -------
        Dict[str, str]
            Dictionary with 'instruction', 'examples', 'constraints' keys.
        """
        prompt_text = self._extract_prompt_text(prompt)

        parts = {
            "instruction": "",
            "examples": "",
            "constraints": "",
        }

        # Simple heuristic: split by common patterns
        lines = prompt_text.split('\n')
        current_section = None
        section_content = []

        for line in lines:
            line_lower = line.lower()

            # Detect section headers
            if line_lower.startswith("instruction:"):
                current_section = "instruction"
                continue

            if line_lower.startswith("example:"):
                current_section = "example"
                continue

            if line_lower.startswith("constraint:"):
                current_section = "constraint"
                continue

            # Accum content
            if current_section:
                section_content[current_section].append(line)
            elif current_section:
                # Fuzzy match
                if "instruction" in line or "constraint" in line:
                    parts["instruction"] += "\n" + line
                elif "example" in line_lower or "here's an example" in line_lower:
                    parts["examples"] += "\n" + line
                elif "constraint" in line.lower:
                    parts["constraints"] += "\n" + line
                else:
                    parts["instruction"] += "\n" + line

        # Clean up sections
        for section in parts:
            parts[section] = parts[section].strip()

        return parts

    def _build_section_rewrite_prompt(
        self,
        current_content: str,
        section: str,
        experiences: List[Experience],
    ) -> str:
        """Build a prompt to rewrite a specific section.

        Parameters
        ----------
        current_content : str
            Current section content.
        section : str
            Which section to rewrite.
        experiences : List[Experience]
            Experiences to learn from.

        Returns
        -------
        str
            Rewrite content for the section.
        """
        # Extract prompt text
        prompt_text = self._extract_prompt_text(
            OptimizablePrompt(content=current_content) if not isinstance(current_content, str) else current_content
        )

        # Build rewrite request
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert prompt engineer. Rewrite ONLY the {section} section "
                    f"of the given system prompt. Keep the style and tone consistent "
                    f"with the original. Return ONLY the rewritten {section} text. "
                    f"No commentary, no explanations, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current {section}:\n\"\"\"\n{current_content}\n\"\"\"\n\n"
                    f"Failure experiences:\n{self._format_experiences_for_section(experiences, section)}\n\n"
                    f"Rewrite the {section} to improve performance."
                ),
            },
        ]

        # Call LLM
        new_content = self.adapter._call_api(
            messages=rewrite_messages,
            system=None,
            temperature=0.5,
        )
        new_content = new_content.strip()

        # Clean up markdown fences
        if new_content.startswith("```"):
            lines = new_content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_content = "\n".join(lines).strip()

        return new_content

    def _assemble_prompt_sections(self, parts: Dict[str, str]) -> str:
        """Assemble prompt sections back into full prompt.

        Parameters
        ----------
        parts : Dict[str, str]
            Dictionary with section content.

        Returns
        -------
        str
            Full prompt text.
        """
        # Reconstruct in order
        sections = ["instruction", "examples", "constraints"]
        content_parts = []

        for section in sections:
            if section in parts:
                content_parts.append(parts[section])

        # Join with newlines
        return "\n\n".join(content_parts)

    def _format_experiences_for_section(
        self, experiences: List[Experience], section: str
    ) -> str:
        """Format experiences for section-specific optimization.

        Parameters
        ----------
        experiences : List[Experience]
            List of experiences.
        section : str
            Which section is being optimized.

        Returns
        -------
        str
            Formatted experience descriptions.
        """
        formatted = []

        for exp in experiences:
            feedback = exp.get_feedback()
            if not feedback:
                continue

            # Extract input
            user_input = exp.get_input()
            if isinstance(user_input, list):
                user_text = user_input[-1].get("content", "") if user_input else ""
            elif isinstance(user_input, dict):
                user_text = user_input.get("text", str(user_input))
            else:
                user_text = str(user_input)

            # Extract output
            agent_response = exp.get_output()
            if isinstance(agent_response, dict):
                agent_text = agent_response.get("text", str(agent_response))
            else:
                agent_text = str(agent_response)

            # Format feedback
            if hasattr(feedback, 'to_score'):
                score = feedback.to_score()
                feedback_text = f"Score: {score:.2f}"
            elif hasattr(feedback, 'critique') and feedback.critique:
                feedback_text = f"Critique: {feedback.critique}"
            elif hasattr(feedback, 'correction') and feedback.correction:
                feedback_text = f"Correction: {feedback.correction}"
            else:
                feedback_text = "Feedback provided"

            formatted.append(
                f"- User: \"{user_text[:80]}...\"\n"
                f"  Agent: \"{agent_text[:80]}...\"\n"
                f"  Feedback: {feedback_text}"
            )

        return "\n".join(formatted)

    # ------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------

    def _walk_tree(self, tree: Any, order: str = "bottom_up") -> List[tuple]:
        """Walk the tree in specified order.

        Parameters
        ----------
        tree : SkillTree
            The skill tree.
        order : str
            'bottom_up' or 'top_down'.

        Returns
        -------
        List[tuple]
            List of (path, node) tuples.
        """
        nodes = []

        def _collect(node, path=""):
            current_path = path
            nodes.append((current_path, node))

            for child_name, child_node in node.children.items():
                child_path = f"{path}.{child_name}" if path else child_name
                _collect(child_node, child_path)

        _collect(tree.root)

        if order == "bottom_up":
            # Reverse to get leaves first
            nodes.reverse()

        return nodes

    def _optimize_node(
        self,
        node: Any,
        experiences: List[Experience],
        validator: Optional[Validator],
        section: Optional[str] = None,
    ) -> Optional[OptimizablePrompt]:
        """Optimize a single node's prompt.

        Parameters
        ----------
        node : SkillNode
            The node to optimize.
        experiences : List[Experience]
            Experiences with feedback.
        validator : Optional[Validator]
            Optional validator.
        section : Optional[str]
            Which section to optimize (for section-wise optimization).

        Returns
        -------
        Optional[OptimizablePrompt]
            Optimized prompt or None.
        """
        # Get node's prompt
        prompt = self._get_node_prompt(node)

        if not prompt:
            logger.warning(f"No prompt found for node: {node.name}")
            return None

        # Use section-wise optimization or regular optimization
        if section and section != "all":
            return self.optimize_prompt_section(
                prompt=prompt,
                experiences=experiences,
                section=section,
            )
        else:
            result = self.base_optimizer.optimize(
                prompt=prompt,
                experiences=experiences,
                validator=validator,
            )
            return result.optimized_prompt

    def _get_node_prompt(self, node: Any) -> Optional[OptimizablePrompt]:
        """Extract an OptimizablePrompt from a SkillNode."""
        from evoskill.core.prompts import TextPrompt

        if hasattr(node.skill, 'system_prompt'):
            return TextPrompt(content=node.skill.system_prompt)
        elif hasattr(node.skill, 'prompt'):
            return TextPrompt(content=node.skill.prompt)
        else:
            return None

    def _extract_prompt_text(self, prompt: OptimizablePrompt) -> str:
        """Extract text from an OptimizablePrompt."""
        # Handle plain string values.
        if isinstance(prompt, str):
            return prompt

        # Handle object-like values.
        if hasattr(prompt, "content"):
            return prompt.content
        elif hasattr(prompt, "text"):
            return prompt.text
        elif hasattr(prompt, "instruction"):
            return prompt.instruction
        elif hasattr(prompt, "system_prompt"):
            return prompt.system_prompt
        elif hasattr(prompt, "to_model_input"):
            return str(prompt.to_model_input())
        else:
            # Final fallback.
            return str(prompt)

    def _create_skill_from_prompt(self, prompt: OptimizablePrompt, template_skill: Any) -> Any:
        """Create a Skill from an OptimizablePrompt using a template."""
        prompt_text = self._extract_prompt_text(prompt)

        # Copy template skill with new prompt
        new_skill = template_skill.model_copy(
            update={
                "system_prompt": prompt_text,
                "version": "v1.0",
            }
        )
        return new_skill

"""
Tree-Aware Optimization Example

This example demonstrates how to use TreeAwareOptimizer to:
1. Load a skill tree
2. Collect experiences with feedback
3. Run tree-aware optimization with automatic splitting and pruning
4. Save the optimized tree

```mermaid
graph TD
    A[Load Skill Tree] --> B[Collect Experiences]
    B --> C[Create TreeAwareOptimizer]
    C --> D[Run Optimization]
    D --> E{Analyze Each Node}
    E --> F[Single-Point Optimization]
    F --> G{Need Split?}
    G -->|Yes| H[Split into Children]
    G -->|No| I{Need Prune?}
    H --> I
    I -->|Yes| J[Prune Node]
    I -->|No| K[Keep Node]
    J --> L[Save Optimized Tree]
    K --> L
```
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treeskill import (
    SkillTree,
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load or Create Skill Tree
# ---------------------------------------------------------------------------

def load_or_create_skill_tree() -> SkillTree:
    """Load existing tree or create a new one.

    ```mermaid
    graph LR
        A[Check if tree exists] -->|Yes| B[Load from disk]
        A -->|No| C[Create new tree]
        C --> D[Save to disk]
        B --> E[Return tree]
        D --> E
    ```
    """
    tree_path = Path("example-skills/")

    if tree_path.exists():
        logger.info(f"📂 Loading existing skill tree from {tree_path}")
        tree = SkillTree.load(tree_path)
    else:
        logger.info(f"🆕 Creating new skill tree at {tree_path}")
        # Create a simple tree with root skill
        from treeskill import Skill, SkillMeta

        root_skill = Skill(
            name="assistant",
            system_prompt="You are a helpful assistant. Help users with various writing tasks.",
            version="v1.0",
        )

        tree = SkillTree(
            root=SkillNode(name="assistant", skill=root_skill),
            base_path=tree_path,
        )
        tree.save()

    logger.info(f"✅ Tree loaded:\n{tree.list_tree()}")
    return tree


# ---------------------------------------------------------------------------
# Step 2: Collect Experiences with Feedback
# ---------------------------------------------------------------------------

def collect_experiences() -> List[ConversationExperience]:
    """Collect experiences with feedback from user interactions.

    ```mermaid
    graph TD
        A[User Interacts] --> B[Generate Response]
        B --> C[User Provides Feedback]
        C --> D{Feedback Type}
        D -->|Positive| E[Record Success]
        D -->|Negative| F[Record Failure]
        D -->|Correction| G[Record Ideal Response]
        E --> H[Store Experience]
        F --> H
        G --> H
    ```
    """
    logger.info("📊 Collecting experiences with feedback...")

    experiences = []

    # Example 1: Formal email task - Good response
    exp1 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Write a formal business email to a client"},
            {"role": "assistant", "content": "Dear Client, I hope this email finds you well..."},
        ]
    )
    exp1.feedback = CompositeFeedback(
        feedback_type=FeedbackType.POSITIVE,
        critique="Appropriate formal tone",
        score=0.9,
    )
    experiences.append(exp1)

    # Example 2: Casual chat task - Bad response (too formal)
    exp2 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Chat with me casually about hobbies"},
            {"role": "assistant", "content": "Dear Sir/Madam, I would be delighted to discuss..."},
        ]
    )
    exp2.feedback = CompositeFeedback(
        feedback_type=FeedbackType.NEGATIVE,
        critique="Too formal for casual chat",
        correction="Hey! I'd love to chat about hobbies! What do you enjoy doing?",
        score=0.2,
    )
    experiences.append(exp2)

    # Example 3: Another formal task - Good
    exp3 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Write a professional report summary"},
            {"role": "assistant", "content": "Executive Summary: This report outlines..."},
        ]
    )
    exp3.feedback = CompositeFeedback(
        feedback_type=FeedbackType.POSITIVE,
        critique="Professional and clear",
        score=0.85,
    )
    experiences.append(exp3)

    # Example 4: Another casual task - Bad (too formal again)
    exp4 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Help me write a fun social media post"},
            {"role": "assistant", "content": "Dear Valued Followers, It is with great pleasure..."},
        ]
    )
    exp4.feedback = CompositeFeedback(
        feedback_type=FeedbackType.NEGATIVE,
        critique="Way too formal for social media",
        correction="Hey friends! 🎉 Check this out!",
        score=0.15,
    )
    experiences.append(exp4)

    # Example 5: More formal tasks
    exp5 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Draft a formal apology letter"},
            {"role": "assistant", "content": "Dear Mr. Smith, I am writing to sincerely apologize..."},
        ]
    )
    exp5.feedback = CompositeFeedback(
        feedback_type=FeedbackType.POSITIVE,
        critique="Perfect formal tone for apology",
        score=0.88,
    )
    experiences.append(exp5)

    # Example 6: Another casual task - Better response
    exp6 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Write a friendly reminder to my team"},
            {"role": "assistant", "content": "Hey team! Just a quick reminder about..."},
        ]
    )
    exp6.feedback = CompositeFeedback(
        feedback_type=FeedbackType.POSITIVE,
        critique="Good casual tone",
        score=0.75,
    )
    experiences.append(exp6)

    # Example 7: One more formal
    exp7 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Create a formal meeting invitation"},
            {"role": "assistant", "content": "You are cordially invited to attend..."},
        ]
    )
    exp7.feedback = CompositeFeedback(
        feedback_type=FeedbackType.POSITIVE,
        critique="Professional invitation",
        score=0.9,
    )
    experiences.append(exp7)

    logger.info(f"✅ Collected {len(experiences)} experiences")
    logger.info(f"   - Positive: {sum(1 for e in experiences if e.feedback.to_score() >= 0.6)}")
    logger.info(f"   - Negative: {sum(1 for e in experiences if e.feedback.to_score() < 0.6)}")

    return experiences


# ---------------------------------------------------------------------------
# Step 3: Create and Configure TreeAwareOptimizer
# ---------------------------------------------------------------------------

def create_optimizer():
    """Create TreeAwareOptimizer with configuration.

    ```mermaid
    graph LR
        A[Create Adapter] --> B[Configure Optimizer]
        B --> C[Create TreeAwareOptimizer]
        C --> D[Return]
    ```
    """
    logger.info("⚙️  Creating TreeAwareOptimizer...")

    # Create adapter (use OpenAI or Anthropic)
    # Note: You need to set OPENAI_API_KEY environment variable
    adapter = OpenAIAdapter(model="gpt-4o-mini")

    # Configure tree optimization
    config = TreeOptimizerConfig(
        auto_split=True,           # Enable automatic splitting
        auto_prune=True,           # Enable automatic pruning
        prune_threshold=0.3,       # Prune nodes with performance < 30%
        min_samples_for_split=5,   # Need at least 5 samples to consider split
        max_tree_depth=3,          # Maximum tree depth
        optimization_order="bottom_up",  # Optimize leaves first
        section="all",             # Optimize entire prompts
    )

    # Create optimizer
    tree_optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
    )

    logger.info("✅ Optimizer created")
    logger.info(f"   - Auto-split: {config.auto_split}")
    logger.info(f"   - Auto-prune: {config.auto_prune}")
    logger.info(f"   - Prune threshold: {config.prune_threshold}")

    return tree_optimizer


# ---------------------------------------------------------------------------
# Step 4: Run Optimization
# ---------------------------------------------------------------------------

def run_optimization(tree: SkillTree, experiences: List[ConversationExperience]):
    """Run tree optimization.

    ```mermaid
    graph TD
        A[Start Optimization] --> B[Walk Tree Bottom-Up]
        B --> C{For Each Node}
        C --> D[Optimize Node]
        D --> E{Check Split Need}
        E -->|Contradictory Feedback| F[Analyze Split]
        F --> G[Generate Children]
        G --> H[Add to Tree]
        E -->|No Split| I{Check Prune Need}
        I -->|Low Performance| J[Prune Node]
        I -->|Good Performance| K[Keep Node]
        H --> C
        J --> C
        K --> C
        C -->|Done| L[Return Result]
    ```
    """
    logger.info("\n" + "="*60)
    logger.info("🌳 Starting Tree-Aware Optimization")
    logger.info("="*60)

    tree_optimizer = create_optimizer()

    # Run optimization
    result = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=experiences,
        validator=None,  # Optional: add custom validator
    )

    logger.info("\n" + "="*60)
    logger.info("✅ Optimization Complete!")
    logger.info("="*60)
    logger.info(f"📊 Results:")
    logger.info(f"   - Nodes optimized: {result.nodes_optimized}")
    logger.info(f"   - Splits performed: {result.splits_performed}")
    logger.info(f"   - Nodes pruned: {result.prunes_performed}")

    return result


# ---------------------------------------------------------------------------
# Step 5: Save and Inspect Results
# ---------------------------------------------------------------------------

def save_and_inspect(tree: SkillTree, result: TreeOptimizationResult):
    """Save optimized tree and inspect results.

    ```mermaid
    graph LR
        A[Save Tree] --> B[Print Summary]
        B --> C[Show Tree Structure]
        C --> D[Display Metrics]
    ```
    """
    # Save optimized tree
    output_path = Path("example-skills-optimized/")
    tree.save(output_path)
    logger.info(f"\n💾 Saved optimized tree to {output_path}")

    # Display optimized tree structure
    logger.info(f"\n🌲 Optimized Tree Structure:")
    logger.info(f"\n{tree.list_tree()}")

    # Display optimization metrics
    if result.splits_performed > 0:
        logger.info(f"\n✂️  Splits performed: {result.splits_performed}")
        logger.info("   The optimizer detected contradictory feedback and split skills")

    if result.prunes_performed > 0:
        logger.info(f"\n✂️  Nodes pruned: {result.prunes_performed}")
        logger.info("   Low-performing nodes were automatically removed")

    if result.nodes_optimized > 0:
        logger.info(f"\n✨ Nodes optimized: {result.nodes_optimized}")
        logger.info("   Prompts were improved based on feedback")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution flow.

    ```mermaid
    graph TD
        A[Start] --> B[Load Skill Tree]
        B --> C[Collect Experiences]
        C --> D[Run Optimization]
        D --> E[Save Results]
        E --> F[Display Summary]
        F --> G[End]
    ```
    """
    logger.info("🚀 Starting Tree-Aware Optimization Example\n")

    # Step 1: Load or create skill tree
    tree = load_or_create_skill_tree()

    # Step 2: Collect experiences
    experiences = collect_experiences()

    # Step 3-4: Run optimization
    result = run_optimization(tree, experiences)

    # Step 5: Save and inspect
    save_and_inspect(tree, result)

    logger.info("\n✨ Example complete!")
    logger.info("Check the 'example-skills-optimized/' directory to see the optimized tree")


if __name__ == "__main__":
    # Import SkillNode (needed for creating tree)
    from treeskill.skill_tree import SkillNode

    main()

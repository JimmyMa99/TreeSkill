"""Allow ``python -m treeskill`` to launch the primary pipeline entrypoint."""

from treeskill.pipeline_main import main

raise SystemExit(main())

"""JavaScript / TypeScript dependency parser.

Covers .js, .jsx, .ts, .tsx files.
Uses regex — no full JS/TS AST parser dependency.

Recognised patterns:
    import ... from './foo'
    import './foo'
    export ... from './foo'
    const x = require('./foo')
    const x = require("./foo")

Only relative imports (starting with . or /) are captured, as those are
the ones that resolve to intra-project files. Third-party package imports
(e.g. 'react', 'lodash') are ignored.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches:
#   import ... from './foo'          ES module named/default import
#   export ... from './foo'          re-export
#   import './foo'                   side-effect import (no bindings, no 'from')
#   const x = require('./foo')       CommonJS require
_IMPORT_RE = re.compile(
    r"""
    (?:
        (?:import|export)   # import or export keyword
        (?:[\s\S]*?)        # optional bindings (may span multiple tokens)
        from\s*             # 'from' keyword
        |
        (?:import)\s*       # bare side-effect import: import './foo'
        (?=['"])            # immediately followed by a quote (no bindings)
        |
        require\s*\(\s*     # CommonJS: require( ...
    )
    ['"]                    # opening quote
    ([./][^'"]+)            # capture: relative path (starts with . or /)
    ['"]                    # closing quote
    """,
    re.VERBOSE,
)


class JavaScriptParser:
    extensions: list[str] = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

    def extract_dependencies(self, filepath: Path, content: str) -> list[str]:
        """Return relative import paths found in *content*."""
        return _IMPORT_RE.findall(content)

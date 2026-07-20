// Module `crate::a::c` (reached via `mod c;` in src/a/mod.rs -> src/a/c.rs).

// Glob import: no single item, but names the target MODULE `crate::b`.
use crate::b::*;

#[derive(Default)]
pub struct Helper;

impl Helper {
    pub fn touch() {
        // an inline module has a body: it declares NO new file node.
        mod inner_inline {
            pub fn noop() {}
        }
        inner_inline::noop();
    }
}

// Module `crate::b` (reached via `mod b;` in lib.rs -> src/b.rs).

// `self::` from crate::b resolves to `crate::b` itself (its own module namespace) —
// self::ReExported is a within-file reference, not a cross-file edge, so it is NOT
// in edges.json (it resolves to this same node and self-links are dropped).
use self::ReExported as _Alias;

pub struct ReExported;

impl ReExported {
    pub fn go() {}
}

// Module `crate::a` (reached via `mod a;` in lib.rs -> src/a/mod.rs).

mod c; // file-backed -> src/a/c.rs  (module `crate::a::c`)

// `super::` from crate::a resolves to the crate root's child `crate::b`.
use super::b::ReExported;

pub fn use_it() {
    ReExported::go();
}

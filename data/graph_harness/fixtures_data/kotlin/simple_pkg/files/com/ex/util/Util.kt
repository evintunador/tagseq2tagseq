package com.ex.util

// One file declaring TWO top-level symbols, each imported separately elsewhere:
//   com.ex.util.Helper   (a class)
//   com.ex.util.helperFn (a top-level function)
// This is the multi-symbol-per-file case that forces the symbol->file node model.

class Helper {
    fun greet(): String = "hi"
}

fun helperFn(n: Int): Int = n * 2

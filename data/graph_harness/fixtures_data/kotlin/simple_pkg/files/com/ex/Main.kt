package com.ex

import com.ex.util.Helper
import com.ex.util.helperFn
import com.ex.Consts
import com.ex.foo.bar as Baz
import com.ex.*
import kotlin.collections.List

// import com.fake.ShouldBeIgnored  -- inside a line comment, must NOT be detected
/* import com.fake.AlsoIgnored */

class Main {
    fun run() {
        val h = Helper()
        val doubled = helperFn(Consts.MAX)
        val x: Int = Baz
    }
}

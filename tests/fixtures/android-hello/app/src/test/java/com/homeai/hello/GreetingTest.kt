package com.homeai.hello

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/** Every boundary of [greeting], on both sides, and the range check. */
class GreetingTest {
    @Test fun eachBoundary() {
        val expected = mapOf(
            0 to "Good night", 4 to "Good night", 5 to "Good morning", 11 to "Good morning",
            12 to "Good afternoon", 17 to "Good afternoon", 18 to "Good evening", 21 to "Good evening",
            22 to "Good night", 23 to "Good night",
        )
        for ((hour, text) in expected) assertEquals("hour $hour", text, greeting(hour))
    }

    @Test fun hoursOutsideTheDayAreRejected() {
        for (hour in listOf(-1, 24)) {
            assertThrows(IllegalArgumentException::class.java) { greeting(hour) }
        }
    }
}

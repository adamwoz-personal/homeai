package com.homeai.hello

/** The greeting for an hour of the day (0-23). */
fun greeting(hour: Int): String {
    require(hour in 0..23) { "hour must be 0-23, was $hour" }
    return when (hour) {
        in 5..11 -> "Good morning"
        in 12..17 -> "Good afternoon"
        in 18..21 -> "Good evening"
        else -> "Good night"
    }
}

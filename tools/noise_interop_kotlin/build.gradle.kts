// The app will use Signal's maintained fork of noise-java; this proves it matches Python's noiseprotocol.
plugins {
    kotlin("jvm") version "2.2.10"
    application
}

dependencies {
    implementation("org.signal.forks:noise-java:0.1.1")
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21) } }
java { targetCompatibility = JavaVersion.VERSION_21; sourceCompatibility = JavaVersion.VERSION_21 }

application { mainClass.set("NoiseInteropKt") }

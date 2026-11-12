// Copyright (c) 2026 Adam Wosotowsky
// SPDX-License-Identifier: BUSL-1.1
/**
 * The Kotlin half of tools/noise_interop.py (REMOTE_APP_PLAN §5.3, Phase 0).
 *
 * Runs the same fixed-key Noise_NK_25519_ChaChaPoly_SHA256 session as Python with noise-java, then:
 * every byte it produces must equal Python's, it must decrypt Python's bytes, and a flipped byte must be refused.
 * Writes its own vectors so Python can check them the other way round.
 *
 * Usage: gradlew run --args="<python vectors> <kotlin vectors out>"; prints OK or one problem per line.
 */
import com.southernstorm.noise.protocol.CipherState
import com.southernstorm.noise.protocol.HandshakeState
import java.io.File
import java.security.MessageDigest

const val PROTOCOL = "Noise_NK_25519_ChaChaPoly_SHA256"
const val TAG = 16

/** Public test keys: sha256 of a label, exactly as Python computes them. Never a real pairing key. */
fun testKey(name: String): ByteArray =
    MessageDigest.getInstance("SHA-256").digest("homeai noise test $name".toByteArray())

fun ByteArray.hex(): String = joinToString("") { "%02x".format(it) }
fun String.unhex(): ByteArray = chunked(2).map { it.toInt(16).toByte() }.toByteArray()

fun readVectors(file: File): LinkedHashMap<String, ByteArray> {
    val out = LinkedHashMap<String, ByteArray>()
    file.readLines().forEach { line ->
        val key = line.substringBefore(' ')
        out[key] = line.substringAfter(' ', "").unhex()
    }
    return out
}

/** One side of the fixed session: app (initiator) or home (responder), handshake not yet run. */
fun side(initiator: Boolean, prologue: ByteArray): HandshakeState {
    val hs = HandshakeState(PROTOCOL, if (initiator) HandshakeState.INITIATOR else HandshakeState.RESPONDER)
    if (initiator) {
        val home = HandshakeState(PROTOCOL, HandshakeState.RESPONDER).localKeyPair
        home.setPrivateKey(testKey("home_static_priv"), 0)
        val pub = ByteArray(home.publicKeyLength).also { home.getPublicKey(it, 0) }
        hs.remotePublicKey.setPublicKey(pub, 0)
        hs.fixedEphemeralKey.setPrivateKey(testKey("app_ephemeral_priv"), 0)
    } else {
        hs.localKeyPair.setPrivateKey(testKey("home_static_priv"), 0)
        hs.fixedEphemeralKey.setPrivateKey(testKey("home_ephemeral_priv"), 0)
    }
    hs.setPrologue(prologue, 0, prologue.size)
    hs.start()
    return hs
}

fun write(hs: HandshakeState, payload: ByteArray): ByteArray {
    val buf = ByteArray(65535)
    return buf.copyOf(hs.writeMessage(buf, 0, payload, 0, payload.size))
}

fun read(hs: HandshakeState, message: ByteArray): ByteArray {
    val buf = ByteArray(65535)
    return buf.copyOf(hs.readMessage(message, 0, message.size, buf, 0))
}

fun encrypt(cs: CipherState, plain: ByteArray): ByteArray {
    val buf = ByteArray(plain.size + TAG)
    return buf.copyOf(cs.encryptWithAd(null, plain, 0, buf, 0, plain.size))
}

fun decrypt(cs: CipherState, cipher: ByteArray): ByteArray {
    val buf = ByteArray(cipher.size)
    return buf.copyOf(cs.decryptWithAd(null, cipher, 0, buf, 0, cipher.size))
}

fun flip(b: ByteArray): ByteArray = b.copyOf().also { it[it.size - 1] = (it[it.size - 1].toInt() xor 1).toByte() }

/** Kotlin's own bytes for the session, taking the plaintexts from Python's file (what is sent is not under test). */
fun session(py: Map<String, ByteArray>): LinkedHashMap<String, ByteArray> {
    val out = LinkedHashMap<String, ByteArray>()
    val app = side(true, py.getValue("prologue"))
    val home = side(false, py.getValue("prologue"))
    out["prologue"] = py.getValue("prologue")
    val pub = ByteArray(32).also { home.localKeyPair.getPublicKey(it, 0) }
    out["home_static_pub"] = pub
    out["msg1_payload"] = py.getValue("msg1_payload")
    out["msg2_payload"] = py.getValue("msg2_payload")
    out["msg1"] = write(app, out.getValue("msg1_payload"))
    check(read(home, out.getValue("msg1")).contentEquals(out.getValue("msg1_payload"))) { "own msg1 unreadable" }
    out["msg2"] = write(home, out.getValue("msg2_payload"))
    check(read(app, out.getValue("msg2")).contentEquals(out.getValue("msg2_payload"))) { "own msg2 unreadable" }
    out["handshake_hash"] = app.handshakeHash
    val appCs = app.split()
    val homeCs = home.split()
    for (dir in listOf("a2h", "h2a")) {
        val sender = if (dir == "a2h") appCs.sender else homeCs.sender
        var n = 0
        while (py.containsKey("${dir}_${n}_plain")) {
            val plain = py.getValue("${dir}_${n}_plain")
            out["${dir}_${n}_plain"] = plain
            out["${dir}_${n}_cipher"] = encrypt(sender, plain)
            n++
        }
    }
    return out
}

fun problems(py: Map<String, ByteArray>, ours: Map<String, ByteArray>): List<String> {
    val found = mutableListOf<String>()
    for ((key, value) in py) {
        val mine = ours[key]
        if (mine == null) found += "missing $key" else if (!mine.contentEquals(value)) found += "$key differs"
    }
    if (found.isNotEmpty()) return found
    // Kotlin as home reads Python's app, Kotlin as app reads Python's home.
    val app = side(true, py.getValue("prologue"))
    val home = side(false, py.getValue("prologue"))
    write(app, py.getValue("msg1_payload"))
    if (!read(home, py.getValue("msg1")).contentEquals(py.getValue("msg1_payload"))) found += "home could not read msg1"
    write(home, py.getValue("msg2_payload"))
    if (!read(app, py.getValue("msg2")).contentEquals(py.getValue("msg2_payload"))) found += "app could not read msg2"
    val appCs = app.split()
    val homeCs = home.split()
    var n = 0
    while (py.containsKey("a2h_${n}_cipher")) {
        if (!decrypt(homeCs.receiver, py.getValue("a2h_${n}_cipher")).contentEquals(py.getValue("a2h_${n}_plain")))
            found += "home could not decrypt a2h_$n"
        n++
    }
    n = 0
    while (py.containsKey("h2a_${n}_cipher")) {
        if (!decrypt(appCs.receiver, py.getValue("h2a_${n}_cipher")).contentEquals(py.getValue("h2a_${n}_plain")))
            found += "app could not decrypt h2a_$n"
        n++
    }
    try {
        read(side(false, py.getValue("prologue")), flip(py.getValue("msg1")))
        found += "tampered msg1 accepted"
    } catch (e: Exception) { /* refused, as it must be */ }
    val app2 = side(true, py.getValue("prologue"))
    val home2 = side(false, py.getValue("prologue"))
    read(home2, write(app2, py.getValue("msg1_payload")))
    read(app2, write(home2, py.getValue("msg2_payload")))
    val home2Cs = home2.split()
    app2.split()
    try {
        decrypt(home2Cs.receiver, flip(py.getValue("a2h_1_cipher")))
        found += "tampered a2h_1 accepted"
    } catch (e: Exception) { /* refused */ }
    return found
}

fun main(args: Array<String>) {
    require(args.size == 2) { "usage: <python vectors> <kotlin vectors out>" }
    val py = readVectors(File(args[0]))
    val ours = session(py)
    File(args[1]).writeText(ours.entries.joinToString("") { "${it.key} ${it.value.hex()}\n" })
    val found = problems(py, ours)
    println(if (found.isEmpty()) "OK kotlin agrees with python (${py.size} vectors)" else found.joinToString("\n"))
    if (found.isNotEmpty()) kotlin.system.exitProcess(1)
}

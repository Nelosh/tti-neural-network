package lv.tti.neuralnetwork.api.idx

import java.io.Closeable
import java.io.DataInputStream
import java.io.File

class IDX(dataFile: File, labelFile: File) : Iterator<Pair<Int, Array<Int>>>, Closeable {

	companion object {
		const val NUMBERS_MAGIC_NUMBER = 2051
		const val LABEL_MAGIC_NUMBER = 2049
	}

	private val numberBytes: DataInputStream = DataInputStream(dataFile.inputStream())
	private val labelBytes: DataInputStream = DataInputStream(labelFile.inputStream())

	private val numberOfCols: Int
	private val numberOfRows: Int
	private val numbersCount: Int

	init {
		checkMagicNumber(numberBytes, NUMBERS_MAGIC_NUMBER)
		checkMagicNumber(labelBytes, LABEL_MAGIC_NUMBER)
		numbersCount = numberBytes.readInt()
		numberOfRows = numberBytes.readInt()
		numberOfCols = numberBytes.readInt()
	}

	override fun close() {
		numberBytes.close()
		labelBytes.close()
	}

	override fun hasNext() = numberBytes.available() > 0


	override fun next(): Pair<Int, Array<Int>> {
		val label = ubyte(labelBytes.readByte())
		val pixels = Array(numberOfCols * numberOfRows) { ubyte(numberBytes.readByte()) }
		return Pair(label, pixels)
	}

	private fun ubyte(byte: Byte) = byte.toInt() and 0xFF

	private fun checkMagicNumber(dataInputStream: DataInputStream, magicNumber: Int) {
		if (dataInputStream.readInt() != magicNumber) throw RuntimeException("Incorrect magic number")
	}
}
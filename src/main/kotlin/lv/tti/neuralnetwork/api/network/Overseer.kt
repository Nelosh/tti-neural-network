package lv.tti.neuralnetwork.api.network

import golem.*
import golem.matrix.Matrix
import lv.tti.neuralnetwork.api.idx.IDX
import lv.tti.neuralnetwork.api.idx.Resource
import lv.tti.neuralnetwork.api.network.Constants.LABELS_FILE_PATH
import lv.tti.neuralnetwork.api.network.Constants.NUMBERS_FILE_PATH
import lv.tti.neuralnetwork.api.network.Constants.TEST_LABELS_FILE_PATH
import lv.tti.neuralnetwork.api.network.Constants.TEST_NUMBERS_FILE_PATH
import lv.tti.neuralnetwork.api.network.DataHolder.clear
import lv.tti.neuralnetwork.api.network.DataHolder.epoch
import lv.tti.neuralnetwork.api.network.DataHolder.iteration
import lv.tti.neuralnetwork.api.network.DataHolder.iterations
import lv.tti.neuralnetwork.api.network.DataHolder.testRecog
import lv.tti.neuralnetwork.api.network.DataHolder.testScore
import lv.tti.neuralnetwork.api.network.DataHolder.trainRecog
import lv.tti.neuralnetwork.api.network.DataHolder.trainScore
import kotlin.concurrent.thread

object Overseer {
	var trainingData: Pair<Matrix<Double>, Matrix<Double>> = Pair(zeros(0), zeros(0))
	var testData: Pair<Matrix<Double>, Matrix<Double>> = Pair(zeros(0), zeros(0))

	var trainResults = emptyList<Pair<Int, Double>>()
	var testResults = emptyList<Pair<Int, Double>>()

	var shouldStop = false

	fun prepare() {
		val thread1 = daemon { trainingData = prepareData("training", NUMBERS_FILE_PATH, LABELS_FILE_PATH) }
		val thread2 = daemon { testData = prepareData("test", TEST_NUMBERS_FILE_PATH, TEST_LABELS_FILE_PATH) }
		await { thread1.isAlive || thread2.isAlive }
	}

	fun train(learningRate: Double, batchSize: Int, epochLimit: Int, hiddenLayer: Collection<Int>) {
		trainResults = emptyList<Pair<Int, Double>>()
		testResults = emptyList<Pair<Int, Double>>()
		Overseer.shouldStop = false
		clear()

		val network = Network(784, 10, hiddenLayer).with(learningRate)


		for (j in 1..epochLimit) {
			epoch = j

			val trainingSet = trainingData.first.to2DArray().zip(trainingData.second.to2DArray()).shuffled()

			var dataBatch = emptyList<DoubleArray>()
			var labelBatch = emptyList<DoubleArray>()

			var i = 0
			trainingSet.forEach {
				if (shouldStop)
					return
				dataBatch = dataBatch.plusElement(it.first)
				labelBatch = labelBatch.plusElement(it.second)
				if (dataBatch.size >= batchSize) {
					iteration = ++i

					network.train(create(dataBatch.toTypedArray()), create(labelBatch.toTypedArray()))
					dataBatch = emptyList<DoubleArray>()
					labelBatch = emptyList<DoubleArray>()
				}
			}
			if (!dataBatch.isEmpty()) {
				network.train(create(dataBatch.toTypedArray()), create(labelBatch.toTypedArray()))
			}
			val copy = network.copy()
			daemon { plotResults(copy, trainingData, testData) }
			if (testRecog > 90 || testScore < 0.001 || shouldStop)
				return
		}
	}

	private fun prepareData(type: String, dataPath: String, labelPath: String): Pair<Matrix<Double>, Matrix<Double>> {
		var dataList: List<DoubleArray> = emptyList()
		var labelList: List<DoubleArray> = emptyList()

		var i = 0
		val idx = IDX(Resource(dataPath), Resource(labelPath))
		idx.use { numbers ->
			numbers.forEach { number ->
				iterations += Pair(type, ++i)
				dataList = dataList.plusElement(number.second.map { it.toDouble() }.toDoubleArray())
				labelList = labelList.plusElement(List(10) { if (it == number.first) 1.0 else 0.0}.toDoubleArray())
			}
		}
		return Pair(create(dataList.toTypedArray()), create(labelList.toTypedArray()))
	}

	private fun plotResults(network: Network, trainingData: Pair<Matrix<Double>, Matrix<Double>>, testData: Pair<Matrix<Double>, Matrix<Double>>) {
		val thread1 = daemon { trainResults += network.error(trainingData.first, trainingData.second) }
		val thread2 = daemon { testResults += network.error(testData.first, testData.second) }

		val trainingExampleCount = trainingData.first.numRows().toDouble()
		val testExampleCount = testData.first.numRows().toDouble()

		await { thread1.isAlive || thread2.isAlive }

		daemon {
			if (!trainResults.isEmpty()) {
				trainScore = trainResults.last().second
				trainRecog = trainResults.last().first * 100 / trainingExampleCount
			}
			if (!testResults.isEmpty()) {
				testScore = testResults.last().second
				testRecog = testResults.last().first * 100 / testExampleCount
			}
		}

		plotResult(1,  trainResults.map { it.second }, "Train data Cost", "b")
		plotResult(1,  testResults.map { it.second }, "Test data Cost", "r")
		title("Cost function")

		plotResult(2, trainResults.map { it.first * 100 / trainingExampleCount }, "Train data Cost", "b")
		plotResult(2,  testResults.map { it.first * 100 / testExampleCount }, "Test data Cost", "r")
		title("Recognition results")
	}

	private fun plotResult(figure: Int, scores: List<Double>, seriesName: String, color: String) {
		if (figures[figure] != null)
			figures[figure]!!.first.removeSeries(seriesName)
		figure(figure)
		plot(1..scores.size, scores.toDoubleArray(), color, seriesName)
	}

	private fun await(func: () -> Boolean) {
		while (func()) {}
	}

}

fun daemon(block: () -> Unit) = thread(isDaemon = true, block = block)

object DataHolder {
	var iterations: Map<String, Int> = emptyMap()

	var epoch = 0
	var iteration = 0
	var trainScore = Double.POSITIVE_INFINITY
	var trainRecog = 0.0
	var testScore = Double.POSITIVE_INFINITY
	var testRecog = 0.0

	fun clear() {
		iterations = emptyMap()

		epoch = 0
		iteration = 0
		trainScore = Double.POSITIVE_INFINITY
		trainRecog = 0.0
		testScore = Double.POSITIVE_INFINITY
		testRecog = 0.0
	}
}

object Constants {
	const val TRAINING = "training"
	const val TEST = "test"

	const val NUMBERS_FILE_PATH = "/data/train-images.idx3-ubyte"
	const val LABELS_FILE_PATH = "/data/train-labels.idx1-ubyte"
	const val TEST_NUMBERS_FILE_PATH = "/data/t10k-images.idx3-ubyte"
	const val TEST_LABELS_FILE_PATH = "/data/t10k-labels.idx1-ubyte"
}
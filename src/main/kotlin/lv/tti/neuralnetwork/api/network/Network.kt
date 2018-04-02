package lv.tti.neuralnetwork.api.network

import golem.*
import golem.matrix.Matrix

class Network(private val inputSize: Int, private val outputSize: Int, private val hiddenSizes: Collection<Int>) {

	companion object {
		private const val EPSILON = 1.2
	}

	private var weights: List<Matrix<Double>> = initializeWeights()

	private var learningRate = 1.0

	fun train(input: Matrix<Double>, results: Matrix<Double>) {
		val length = weights.size
		//Forward
		val layers = Array(length+1) { zeros(0)}
		val accumulated = Array(length) { zeros(0)}

		layers[0] = input
		(0 until length).forEach {
			accumulated[it] = biased(layers[it]) * weights[it]
			layers[it+1] = sigmoid(accumulated[it])
		}

		//Backward
		val errors = Array(length) { zeros(0)}
		errors[length-1] = (layers[length] - results) emul (sigmoidGradient(accumulated[length-1]))
		(length-2 downTo 0).forEach {
			errors[it] = (errors[it+1] * unbiased(weights[it+1]).T) emul (sigmoidGradient(accumulated[it]))
		}

		//Weight update
		weights = weights.mapIndexed { index, matrix ->
			matrix - learningRate * biased(layers[index]).T * errors[index] / input.numRows().toDouble()
		}
	}

	private fun initializeWeights() =
			(listOf(inputSize) + hiddenSizes)
					.zip(hiddenSizes + outputSize)
					.map { random(it.first, it.second) }

	private fun random(rows: Int, columns: Int) = rand(rows + 1/*bias*/, columns).mapMat { it*2* EPSILON - EPSILON }

	private fun calculate(input: Matrix<Double>) = weights.fold(input) {
		acc, elem -> (biased(acc)*elem).mapMat(::sigmoid)
	}

	fun error(input: Matrix<Double>, expectedResults: Matrix<Double>): Pair<Int, Double> {
		val calculated = calculate(input)

		val correct = correct(calculated, expectedResults)
		val error = cost(calculated, expectedResults)
		return Pair(correct, error)
	}


	fun copy(): Network {
		val copy = Network(this.inputSize, this.outputSize, this.hiddenSizes)
		copy.weights = this.weights.map { create(it.to2DArray()) }
		return copy
	}

	fun with(learningRate: Double): Network {
		this.learningRate = learningRate
		return this
	}

	private fun cost(input: Matrix<Double>, expectedResults: Matrix<Double>): Double =
			-(expectedResults.emul(input.mapMat(::safeLog)) +
					(1.0 - expectedResults).emul((1.0 - input).mapMat(::safeLog))).mean()

	private fun correct(input: Matrix<Double>, expectedResults: Matrix<Double>): Int =
			input.to2DArray().zip(expectedResults.to2DArray()).sumBy {
				val calculatedIndex = best(it.first)
				val expectedIndex = best(it.second)
				if (calculatedIndex == expectedIndex) 1 else 0
			}

	private fun best(collection: DoubleArray) = collection.mapIndexed { i, elem -> Pair(i, elem)}.maxBy { it.second }!!.first


	private fun safeLog(num: Double) = if (abs(num) < 0.000001) log(0.000001) else log(num)

	private fun sigmoid(matrix: Matrix<Double>) = matrix.mapMat(::sigmoid)
	private fun sigmoid(num: Double) = 1.0/(1.0 + exp(-num))

	private fun sigmoidGradient(matrix: Matrix<Double>) = matrix.mapMat(::sigmoidGradient)
	private fun sigmoidGradient(num: Double): Double {
		val sig = sigmoid(num)
		return sig*(1.0-sig)
	}

	private fun biased(matrix: Matrix<Double>) = create(matrix.to2DArray().map { doubleArrayOf(1.0) + it }.toTypedArray())
	private fun unbiased(matrix: Matrix<Double>) = matrix[1 until matrix.numRows(), 0 until matrix.numCols()]
}
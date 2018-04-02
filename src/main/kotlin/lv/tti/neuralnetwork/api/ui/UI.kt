package lv.tti.neuralnetwork.api.ui

import javafx.beans.property.SimpleBooleanProperty
import javafx.beans.property.SimpleDoubleProperty
import javafx.beans.property.SimpleIntegerProperty
import javafx.beans.property.SimpleStringProperty
import lv.tti.neuralnetwork.api.network.Constants.TEST
import lv.tti.neuralnetwork.api.network.Constants.TRAINING
import lv.tti.neuralnetwork.api.network.DataHolder
import lv.tti.neuralnetwork.api.network.DataHolder.iterations
import lv.tti.neuralnetwork.api.network.Overseer
import lv.tti.neuralnetwork.api.network.daemon
import lv.tti.neuralnetwork.api.ui.Model.batchSize
import lv.tti.neuralnetwork.api.ui.Model.epochInfo
import lv.tti.neuralnetwork.api.ui.Model.epochLimit
import lv.tti.neuralnetwork.api.ui.Model.hiddenLayer
import lv.tti.neuralnetwork.api.ui.Model.iterationInfo
import lv.tti.neuralnetwork.api.ui.Model.learningRate
import lv.tti.neuralnetwork.api.ui.Model.testPrepLabel
import lv.tti.neuralnetwork.api.ui.Model.testRecognInfo
import lv.tti.neuralnetwork.api.ui.Model.testScoreInfo
import lv.tti.neuralnetwork.api.ui.Model.trainPrepLabel
import lv.tti.neuralnetwork.api.ui.Model.trainRecognInfo
import lv.tti.neuralnetwork.api.ui.Model.trainScoreInfo
import tornadofx.*
import kotlin.concurrent.thread

class UI : View() {
	var unprepared = SimpleBooleanProperty(true)

	override val root = vbox {
		hbox(10) {
			button("Prepare") {
				useMaxHeight=true
				setOnAction {
					unprepared.set(true)
					val thread = daemon { Overseer.prepare() }

					runAsync(daemon = true) {
						while (thread.isAlive) {
							var trainMessage = ""
							var testMessage = ""
							runAsync(daemon = true) {
								trainMessage = "Training data: " + iterations[TRAINING]
								testMessage = "Test data: " + iterations[TEST]
							} ui {
								trainPrepLabel.set(trainMessage)
								testPrepLabel.set(testMessage)
							}
						}
						unprepared.set(false)
					}
				}
			}
			vbox {
				label(trainPrepLabel)
				label(testPrepLabel)
			}
		}
		hbox {
			label("Input layer: ")
			textfield("784") {
				maxWidth = 50.0
				isDisable=true
			}
			label("Hidden layer: ")
			textfield(hiddenLayer)
			label("Output layer:")
			textfield("10") {
				maxWidth = 50.0
				isDisable=true
			}
		}
		hbox {
			label("Batch size:")
			textfield(batchSize)
			label("Learning rate:")
			textfield(learningRate)
			label("Epoch limit:")
			textfield(epochLimit)
		}
		button("train") {
			disableProperty().bind(unprepared)
			setOnAction {
				unprepared.set(true)
				val thread = daemon {
					Overseer.train(learningRate.value,
								   batchSize.value,
								   epochLimit.value,
								   hiddenLayer.value!!.split(" ").map { it.toInt() })
					unprepared.set(false)
				}
				val modal = find(TrainWindow::class).openModal()
				modal!!.setOnCloseRequest { Overseer.shouldStop = true }
				runAsync(daemon = true) {
					while (thread.isAlive) {
						var epoch = ""
						var iteration = ""
						var trainScore = ""
						var trainRecognized = ""
						var testScore = ""
						var testRecognized = ""
						runAsync(daemon = true) {
							epoch = "Epoch: " + DataHolder.epoch
							iteration = "Iteration: " + DataHolder.iteration
							trainScore = "Training cost: " + DataHolder.trainScore
							trainRecognized = "Training recognized: " + DataHolder.trainRecog
							testScore = "Test cost: " + DataHolder.testScore
							testRecognized = "Test recognized: " + DataHolder.testRecog
						} ui {
							epochInfo.set(epoch)
							iterationInfo.set(iteration)
							trainScoreInfo.set(trainScore)
							trainRecognInfo.set(trainRecognized)
							testScoreInfo.set(testScore)
							testRecognInfo.set(testRecognized)
						}
					}
				}



			}
		}
	}

	class TrainWindow: Fragment() {
		override val root = vbox {
			minWidth = 200.0
			label(Model.epochInfo)
			label(Model.iterationInfo)
			label(Model.trainScoreInfo)
			label(Model.trainRecognInfo)
			label(Model.testScoreInfo)
			label(Model.testRecognInfo)
		}
	}
}

object Model {
	var epochInfo = SimpleStringProperty("Epoch: 0" )
	var iterationInfo = SimpleStringProperty("Iteration: 0")
	var trainScoreInfo = SimpleStringProperty("Training cost: 0")
	var trainRecognInfo = SimpleStringProperty("Training recognized: 0")
	var testScoreInfo = SimpleStringProperty("Test cost: 0")
	var testRecognInfo = SimpleStringProperty("Test recognized: 0")

	val trainPrepLabel = SimpleStringProperty("Training data: 0")
	val testPrepLabel = SimpleStringProperty("Test data: 0")

	val hiddenLayer = SimpleStringProperty("100 10")

	val batchSize = SimpleIntegerProperty(100)
	val learningRate = SimpleDoubleProperty(1.0)
	val epochLimit = SimpleIntegerProperty(100)
}


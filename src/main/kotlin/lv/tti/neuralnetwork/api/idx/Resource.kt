package lv.tti.neuralnetwork.api.idx

import java.io.File

class Resource(private val resourcePath: String) : File(Resource::class.java.getResource(resourcePath).toURI())

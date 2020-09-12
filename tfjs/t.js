import * as tf from '@tensorflow/tfjs';

const MODEL_URL = './model.json';

const model = await tf.loadGraphModel(MODEL_URL);
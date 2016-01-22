/*
 * PredictionResults.cpp
 *
 *  Created on: Jan 21, 2016
 *      Author: giannis
 */

#include "PredictionResults.h"

using namespace std;

namespace mai {



mai::PredictionResults::PredictionResults(const char* predictionResult, double fResultValue)
  : m_predictionResult(predictionResult), m_fResultValue(fResultValue)
{}

mai::PredictionResults::~PredictionResults() {
}

} /* namespace mai */

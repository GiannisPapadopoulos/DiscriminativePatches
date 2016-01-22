/*
 * PredictionResults.h
 *
 *  Created on: Jan 21, 2016
 *      Author: giannis
 */

#ifndef IMAGECATALOG_PREDICTIONRESULTS_H_
#define IMAGECATALOG_PREDICTIONRESULTS_H_

using namespace std;

#include <string>

namespace mai {

class PredictionResults {
 public:
  PredictionResults(const char* predictionResult, double fResultValue);
  virtual ~PredictionResults();

  std::string getPrediction() {
    return m_predictionResult;
  }

  double getResultValue() {
    return m_fResultValue;
  }

  void setPredictionResult(const char* predictionResult) {
    m_predictionResult = predictionResult;
  }

  void setResultValue(double fResultValue)  {
    m_fResultValue = fResultValue;
  }

 private:
   std::string m_predictionResult;
   double m_fResultValue;
};

} /* namespace mai */

#endif /* IMAGECATALOG_PREDICTIONRESULTS_H_ */

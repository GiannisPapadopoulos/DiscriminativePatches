/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * cvHOG.h
 *
 *  Created on: Nov 18, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#include <iostream>
#include <string>

#include "configuration/Configuration.h"
#include "imageCatalog/CatalogClassificationSVM.h"
#include "imageCatalog/CatalogTraining.h"

using namespace std;
using namespace mai;

/**
 * This is just the beginning ..
 */
int main(int argc, char** argv )
{
	if(argc < 2)
	{
		cerr << "The application has 3 operating modes defined in the configuration file:" << endl
			<< "\t1. Train SVMs on a categorized image cataloge." << endl
			<< "\t2. Retrain SVMs on a categorized image cataloge." << endl
			<< "\t3. Classify images using trained SVMs." << endl
			<< "Usage:" << endl
			<< argv[0] << " [options]" << endl
			<< "options:" << endl
			<< "\t-config <config file>\tname and path to a configuration file defining application settings." << endl;
		return -1;
	}

	string strConfigFile = "";

	for(int i = 0; i < argc; ++i)
	{
		if(string(argv[i]) == "-config")
		{
			if(i+1 < argc)
			{
				strConfigFile = string(argv[i+1]);
			}
			else
			{
				cerr << "ERROR: Option -config given without filepath." << endl;
				return -1;
			}
		}
	}

	if(strConfigFile.empty())
	{
		cerr << "ERROR: No configuration file given." << endl;
		return -1;
	}

	Configuration* config = new Configuration(strConfigFile);

	if(config->getApplicationMode() == Configuration::appMode::Undef)
	{
		cout << "ERROR: Application mode undefined. What should I do ?" << endl;
		return -1;
	}

	if(config->getApplicationMode() == Configuration::appMode::Train
			|| config->getApplicationMode() == Configuration::appMode::Retrain)
	{
		cout << "[Main] Training classifiers according to configuration given in " << strConfigFile << endl;
		CatalogTraining* trainer = new CatalogTraining(config);
		trainer->processPipeline();
		delete trainer;
	}

	if(config->getApplicationMode() == Configuration::appMode::Predict)
	{
		cout << "[Main] Predicting image according to configuration given in " << strConfigFile << endl;
		CatalogClassificationSVM* classifier = new CatalogClassificationSVM(config);
		classifier->loadAndPredict();
		delete classifier;
	}

	delete config;

	return 0;
}


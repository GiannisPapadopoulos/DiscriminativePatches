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

#include "Configuration.h"
#include "svm/CatalogueTraining.h"
#include "svm/svmtest.h"

using namespace std;
using namespace mai;

/**
 * This is just the beginning ..
 */
int main(int argc, char** argv )
{
	if(argc < 2)
	{
		cerr << "Train svms on categorized image catalogue." << endl
			<< "Usage:" << endl
			<< argv[0] << " [options]" << endl
			<< "options:" << endl
			<< "\t-config <config file>\tname and path to a configuration file defining application settings." << endl;
		return -1;
	}

	string strConfigFile;

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

		if(string(argv[i]) == "-svmtest")
		{
			if(i+1 < argc)
			{
				int iTest = atoi(argv[i+1]);

				if(iTest == 1)
				{
					SVMTest::test1();
				}
				else if (iTest == 2)
				{
					SVMTest::test2();
				}
				else
				{
					SVMTest::testGeneratedTestData();
				}

				return 0;
			}
			else
			{
				cerr << "ERROR: Option -svmtest given without number." << endl;
				return -1;
			}
		}
	}

	Configuration* config = new Configuration(strConfigFile);

	CatalogueTraining* main = new CatalogueTraining(config);
	main->processPipeline();
	delete main;

	return 0;
}


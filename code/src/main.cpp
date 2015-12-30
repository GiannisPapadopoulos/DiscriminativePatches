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

#include "UDoMLDP.h"
#include "CatalogueDetection.h"
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
		cerr << "Usage:" << endl
			<< argv[0] << " [options]" << endl
			<< "options:" << endl
			<< "\t-catalogue <file path>\tTrain svms on categorized data catalogue. File path is base for subfolders containing data. Folder names are trained categories."
			<< "\t-pos <file path>" << endl
			<< "\t-neg <file path>" << endl;
		return -1;
	}

	string strFilepath;

	string strFilepathPositives;
	string strFilepathNegatives;

	for(int i = 0; i < argc; ++i)
	{
		if(string(argv[i]) == "-catalogue")
		{
			if(i+1 < argc)
			{
				strFilepath = string(argv[i+1]);

				CatalogueDetection* main = new CatalogueDetection(strFilepath);
				main->processPipeline();
				delete main;

				return 0;
			}
			else
			{
				cerr << "ERROR: Option -catalogue given without filepath." << endl;
				return -1;
			}
		}

		if(string(argv[i]) == "-pos")
		{
			if(i+1 < argc)
			{
				strFilepathPositives = string(argv[i+1]);
			}
			else
			{
				cerr << "ERROR: Option -pos given without filepath." << endl;
				return -1;
			}
		}

		if(string(argv[i]) == "-neg")
		{
			if(i+1 < argc)
			{
				strFilepathNegatives = string(argv[i+1]);
			}
			else
			{
				cerr << "ERROR: Option -neg given without filepath." << endl;
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

	//strFilepathPositives = "/home/stefan/Documents/AI/project_1/bsps";
	//strFilepathNegatives = "/home/stefan/Documents/AI/project_1/bsps";


	// Use EITHER this one for testing
	//UDoMLDP::unsupervisedDiscovery(strFilepathPositives, strFilepathNegatives);


	// OR this one
	UDoMLDP* main = new UDoMLDP();
	main->basicDetecion(strFilepathPositives, strFilepathNegatives);
	delete main;

	return 0;
}


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
			<< "\t-pos <file path>" << endl
			<< "\t-neg <file path>" << endl;
		return -1;
	}

	string strFilepathPositives;
	string strFilepathNegatives;

	for(int i = 0; i < argc; ++i)
	{
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


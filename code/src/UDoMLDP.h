/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * UDoMLDP.h
 *
 *  Created on: Oct 25, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef SRC_UDOMLDP_H_
#define SRC_UDOMLDP_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace mai{

/**
 * Unsupervised discovery of mid-level discriminative patches
 */
class UDoMLDP
{
public:
	/**
	 * Initializes object
	 */
	UDoMLDP();

	/**
	 * Deletes something
	 */
	virtual ~UDoMLDP();

	/**
	 * Main algorithm
	 */
	void UnsupervisedDiscovery( std::string &strFilePathPositives, std::string &strFilePathNegatives );

};

}// namespace mai




#endif /* SRC_UDOMLDP_H_ */

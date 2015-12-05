/*****************************************************************************
 * Master AI project ws15 group 4
 * Mid-level discriminative patches
 *
 * svmtest.h
 *
 *  Created on: Nov 24, 2015
 *      Author: stefan
 *
 *****************************************************************************
 * THIS SOFTWARE IS PROVIDED 'AS IS'. ANY WARRANTIES ARE DISCLAIMED. IN
 * NO CASE SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES.
 *****************************************************************************/

#ifndef _SVMTEST_H
#define _SVMTEST_H


#include <string>


namespace mai{

/**
 * Doxygen comments
 * Class description
 */
class SVMTest
{
public:

	/**
	 * Deletes something
	 */
	virtual ~SVMTest();

	/**
	 * test method
	 */
	static void testGeneratedTestData();

	static void test1();

	static void test2();

private:
	/**
	 * Initializes object
	 */
	SVMTest();

};

}// namespace mai

#endif // Include guard


/*
 * ProcessIO.h
 *
 *  Created on: Apr 19, 2011
 *      Author: TF
 */

#ifndef PROCESSIO_H_
#define PROCESSIO_H_

// STL
#include <iostream>

// FEM
#include "FEMEnums.h"

namespace FileIO
{
/**
 * Small class to read process information.
 */
class ProcessIO
{
public:
	/**
	 * read process information from various file type:
	 * - boundary condition files
	 * - source term files
	 *
	 * To store the information an object of type ProcessInfo is used (CSourceTerm,
	 * CBoundaryCondition, CInitialCondition and COutput inherit from class ProcessInfo).
	 * @param in_str the input stream
	 * @param pcs_type the process type
	 * @return false, if the process is of type INVALID_PROCESS, else true
	 * \sa enum ProcessType for valid values
	 */
	static bool readProcessInfo (std::istream& in_str, FiniteElement::ProcessType& pcs_type);
};
}

#endif /* PROCESSIO_H_ */

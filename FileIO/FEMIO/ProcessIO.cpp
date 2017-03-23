/*
 * ProcessIO.cpp
 *
 *  Created on: Apr 19, 2011
 *      Author: TF
 */

// STL
#include <sstream>

// FileIO
#include "ProcessIO.h"
#include "readNonBlankLineFromInputStream.h"

namespace FileIO
{
bool ProcessIO::readProcessInfo (std::istream& in_str, FiniteElement::ProcessType& pcs_type)
{
	std::stringstream ss_in (readNonBlankLineFromInputStream (in_str));
	std::string tmp;
	ss_in >> tmp;
	pcs_type = FiniteElement::convertProcessType(tmp);
	if (pcs_type == FiniteElement::INVALID_PROCESS)
		return false;
	else
		return true;
}
}

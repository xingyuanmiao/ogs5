/*
 * \file ProcessInfo.cpp
 *
 *  Created on: Sep 2, 2010
 *      Author: TF
 */

#include "rf_pcs.h"
#include <ProcessInfo.h>

ProcessInfo::ProcessInfo() :
	_pcs_type (FiniteElement::INVALID_PROCESS), _pcs_pv (FiniteElement::INVALID_PV), _pcs (NULL)
{}

ProcessInfo::ProcessInfo (FiniteElement::ProcessType pcs_type, FiniteElement::PrimaryVariable pcs_pv, CRFProcess* pcs) :
	_pcs_type (pcs_type), _pcs_pv (pcs_pv), _pcs (pcs)
{}

void ProcessInfo::setProcessType (FiniteElement::ProcessType pcs_type)
{
	_pcs_type = pcs_type;
}

void ProcessInfo::setProcessPrimaryVariable (FiniteElement::PrimaryVariable pcs_pv)
{
	_pcs_pv = pcs_pv;
}

void ProcessInfo::setProcess (CRFProcess* pcs)
{
	_pcs = pcs;
}

FiniteElement::ProcessType ProcessInfo::getProcessType () const
{
	return _pcs_type;
}

FiniteElement::PrimaryVariable ProcessInfo::getProcessPrimaryVariable () const
{
	return _pcs_pv;
}

int ProcessInfo::getProcessCompVecIndex () const
{
   return _pcs->pcs_component_number;
}

CRFProcess* ProcessInfo::getProcess () const
{
	return _pcs;
}

ProcessInfo::~ProcessInfo()
{}

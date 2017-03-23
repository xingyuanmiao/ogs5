/*
 * generateBCandGLI.cpp
 *
 *  Created on: Mar 1, 2011
 *      Author: TF
 */

// GEO
#include "GEOObjects.h"
#include "ProjectData.h"
#include "SurfaceVec.h"

// FileIO
#include "XmlIO/XmlGmlInterface.h"

#include "problem.h"
Problem* aproblem = NULL;

#include <algorithm>
#include <QString>

int main (int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "Usage: " << argv[0] << " gml-file" << std::endl;
		return -1;
	}
	GEOLIB::GEOObjects* geo_objs (new GEOLIB::GEOObjects);
	ProjectData* project_data (new ProjectData);
	project_data->setGEOObjects (geo_objs);
	std::string schema_name(
	        "/home/fischeth/workspace/OGS-FirstFloor/sources/FileIO/OpenGeoSysGLI.xsd");
	FileIO::XmlGmlInterface xml(project_data, schema_name);
	std::string fname (argv[1]);
	xml.readFile(QString::fromStdString (fname));

	std::vector<std::string> geo_names;
	geo_objs->getGeometryNames (geo_names);
	if (geo_names.empty ())
	{
		std::cout << "no geometries found" << std::endl;
		return -1;
	}
	const GEOLIB::SurfaceVec* sfc_vec (geo_objs->getSurfaceVecObj(geo_names[0]));
	if (!sfc_vec)
	{
		std::cout << "could not found surfaces" << std::endl;
		delete project_data;
		return -1;
	}
	const size_t n_sfc (sfc_vec->size());

	std::vector<size_t> sfc_pnt_ids;
	for (size_t k(0); k < n_sfc; k++)
	{
		std::string sfc_name;
		if (sfc_vec->getNameOfElementByID(k, sfc_name))
			if (sfc_name.find ("Terrain") != std::string::npos)
			{
				std::cout << k << ": " << sfc_name << std::endl;
				GEOLIB::Surface const* sfc (sfc_vec->getElementByName(sfc_name));
				const size_t n_triangles (sfc->getNTriangles());
				for (size_t j(0); j < n_triangles; j++)
				{
					GEOLIB::Triangle const* tri ((*sfc)[j]);
					for (size_t i(0); i < 3; i++)
						sfc_pnt_ids.push_back ((*tri)[i]);
				}
			}
	}

	// make entries unique
	std::cout << "make points unique ... " << std::flush;
	std::sort (sfc_pnt_ids.begin(), sfc_pnt_ids.end());
	std::vector<size_t>::iterator it (sfc_pnt_ids.begin());
	while (it != sfc_pnt_ids.end())
	{
		std::vector<size_t>::iterator next (it);
		next++;
		if (next != sfc_pnt_ids.end())
		{
			if (*it == *next)
				it = sfc_pnt_ids.erase (it);
			else
				it++;
		}
		else
			it++;
	}
	std::cout << "done" << std::endl;

	std::vector<GEOLIB::Point*> const* geo_pnts (geo_objs->getPointVec(geo_names[0]));
	// write gli file and bc file
	std::ofstream gli_out ("TB.gli");
	std::ofstream bc_out ("TB.bc");
	bc_out << "// file generated by " << argv[0] << std::endl;
	if (gli_out && bc_out)
	{
		gli_out << "#POINTS" << std::endl;
		for (size_t k(0); k < sfc_pnt_ids.size(); k++)
		{
			gli_out << k << " " << *((*geo_pnts)[sfc_pnt_ids[k]]) << " $NAME " << k <<
			std::endl;
			// boundary condition
			bc_out << "#BOUNDARY_CONDITION" << std::endl;
			bc_out << "\t$PCS_TYPE" << std::endl << "\t\tGROUNDWATER_FLOW" << std::endl;
			bc_out << "\t$PRIMARY_VARIABLE" << std::endl << "\t\tHEAD" << std::endl;
			bc_out << "\t$GEO_TYPE" << std::endl << "\t\tPOINT " << k << std::endl;
			bc_out << "\t$DIS_TYPE" << std::endl << "\t\tCONSTANT " <<
			(*((*geo_pnts)[sfc_pnt_ids[k]]))[2] << std::endl;
		}
		gli_out << "#STOP" << std::endl;
		bc_out << "#STOP" << std::endl;
		gli_out.close ();
		bc_out.close ();
	}

	delete project_data;
}

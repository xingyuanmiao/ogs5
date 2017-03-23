/*
 * MeshNodesAlongPolyline.h
 *
 *  Created on: Aug 9, 2010
 *      Author: TF
 */

#ifndef MESHNODESALONGPOLYLINE_H_
#define MESHNODESALONGPOLYLINE_H_

// GEOLIB
#include "Polyline.h"

namespace MeshLib
{
// forward declaration
class CFEMesh;

/**
 * This class computes the ids of the mesh nodes along a polyline.
 *
 * The mesh nodes are sorted as follow:
 * [ ... ids of sorted linear nodes ... | ... ids of unsorted higher order nodes ]
 */
class MeshNodesAlongPolyline
{
public:
	MeshNodesAlongPolyline(GEOLIB::Polyline const* const ply, CFEMesh const* mesh, const bool for_s_term = false);
	const std::vector<size_t>& getNodeIDs () const;
	const GEOLIB::Polyline* getPolyline () const;
	size_t getNumberOfLinearNodes () const;
	std::vector<double> const & getDistOfProjNodeFromPlyStart() const;

private:
	const GEOLIB::Polyline* _ply;
	const CFEMesh* _mesh;
	size_t _linear_nodes;
	std::vector<size_t> _msh_node_ids;
	std::vector<double> _dist_of_proj_node_from_ply_start;
    bool for_source_term; // 17.05.2013. WW
};
}

#endif /* MESHNODESALONGPOLYLINE_H_ */

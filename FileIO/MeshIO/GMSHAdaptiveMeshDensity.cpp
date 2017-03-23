/*
 * GMSHAdaptiveMeshDensity.cpp
 *
 *  Created on: Mar 5, 2012
 *      Author: TF
 */

#include <list>

// FileIO
#include "MeshIO/GMSHAdaptiveMeshDensity.h"

// GEOLIB
#include "Polygon.h"

namespace FileIO {

GMSHAdaptiveMeshDensity::GMSHAdaptiveMeshDensity(double pnt_density, double station_density,
				size_t max_pnts_per_leaf) :
	_pnt_density(pnt_density), _station_density(station_density),
	_max_pnts_per_leaf(max_pnts_per_leaf), _quad_tree(NULL)
{
}

GMSHAdaptiveMeshDensity::~GMSHAdaptiveMeshDensity()
{
	delete _quad_tree;
}

void GMSHAdaptiveMeshDensity::init(std::vector<GEOLIB::Point const*> const& pnts)
{
	// *** QuadTree - determining bounding box
#ifndef NDEBUG
	std::cout << "[GMSHAdaptiveMeshDensity::init]" << "\n";
	std::cout << "\tcomputing axis aligned bounding box (2D) for quadtree ... " << std::flush;
#endif
	GEOLIB::Point min(pnts[0]->getData()), max(pnts[0]->getData());
	size_t n_pnts(pnts.size());
	for (size_t k(1); k<n_pnts; k++) {
		for (size_t j(0); j<2; j++)
			if ((*(pnts[k]))[j] < min[j]) min[j] = (*(pnts[k]))[j];
		for (size_t j(0); j<2; j++)
			if ((*(pnts[k]))[j] > max[j]) max[j] = (*(pnts[k]))[j];
	}
	min[2] = 0.0;
	max[2] = 0.0;
#ifndef NDEBUG
	std::cout << "ok" << "\n";
#endif

	// *** QuadTree - create object
#ifndef NDEBUG
	std::cout << "\tcreating quadtree ... " << std::flush;
#endif
	_quad_tree = new GEOLIB::QuadTree<GEOLIB::Point> (min, max, _max_pnts_per_leaf);
#ifndef NDEBUG
	std::cout << "ok" << "\n";
#endif

	// *** QuadTree - insert points
	addPoints(pnts);
}

void GMSHAdaptiveMeshDensity::addPoints(std::vector<GEOLIB::Point const*> const& pnts)
{
	// *** QuadTree - insert points
	const size_t n_pnts(pnts.size());
#ifndef NDEBUG
	std::cout << "\tinserting " << n_pnts << " points into quadtree ... " <<
	std::flush;
#endif
	for (size_t k(0); k < n_pnts; k++)
		_quad_tree->addPoint(pnts[k]);
#ifndef NDEBUG
	std::cout << "ok" << "\n";
#endif
	_quad_tree->balance();
}

double GMSHAdaptiveMeshDensity::getMeshDensityAtPoint(GEOLIB::Point const*const pnt) const
{
	GEOLIB::Point ll, ur;
	_quad_tree->getLeaf(*pnt, ll, ur);
	return _pnt_density * (ur[0] - ll[0]);
}

double GMSHAdaptiveMeshDensity::getMeshDensityAtStation(GEOLIB::Point const*const pnt) const
{
	GEOLIB::Point ll, ur;
	_quad_tree->getLeaf(*pnt, ll, ur);
	return (_station_density * (ur[0] - ll[0]));
}

void GMSHAdaptiveMeshDensity::getSteinerPoints (std::vector<GEOLIB::Point*> & pnts, size_t additional_levels) const
{
	// get Steiner points
	size_t max_depth(0);
	_quad_tree->getMaxDepth(max_depth);

	std::list<GEOLIB::QuadTree<GEOLIB::Point>*> leaf_list;
	_quad_tree->getLeafs(leaf_list);

	for (std::list<GEOLIB::QuadTree<GEOLIB::Point>*>::const_iterator it(leaf_list.begin()); it
					!= leaf_list.end(); it++) {
		if ((*it)->getPoints().empty()) {
			// compute point from square
			GEOLIB::Point ll, ur;
			(*it)->getSquarePoints(ll, ur);
			if ((*it)->getDepth() + additional_levels > max_depth) {
				additional_levels = max_depth - (*it)->getDepth();
			}
			const size_t n_pnts_per_quad_dim (MathLib::fastpow(2, additional_levels));
			const double delta ((ur[0] - ll[0]) / (2 * n_pnts_per_quad_dim));
			for (size_t i(0); i<n_pnts_per_quad_dim; i++) {
				for (size_t j(0); j<n_pnts_per_quad_dim; j++) {
					pnts.push_back(new GEOLIB::Point (ll[0] + (2*i+1) * delta, ll[1] + (2*j+1) * delta, 0.0));
				}
			}

		}
	}
}

#ifndef NDEBUG
void GMSHAdaptiveMeshDensity::getQuadTreeGeometry(std::vector<GEOLIB::Point*> &pnts,
				std::vector<GEOLIB::Polyline*> &plys) const
{
	std::list<GEOLIB::QuadTree<GEOLIB::Point>*> leaf_list;
	_quad_tree->getLeafs(leaf_list);

	for (std::list<GEOLIB::QuadTree<GEOLIB::Point>*>::const_iterator it(leaf_list.begin()); it
		!= leaf_list.end(); it++) {
		// fetch corner points from leaf
		GEOLIB::Point *ll(new GEOLIB::Point), *ur(new GEOLIB::Point);
		(*it)->getSquarePoints(*ll, *ur);
		size_t pnt_offset (pnts.size());
		pnts.push_back(ll);
		pnts.push_back(new GEOLIB::Point((*ur)[0], (*ll)[1], 0.0));
		pnts.push_back(ur);
		pnts.push_back(new GEOLIB::Point((*ll)[0], (*ur)[1], 0.0));
		plys.push_back(new GEOLIB::Polyline(pnts));
		plys[plys.size()-1]->addPoint(pnt_offset);
		plys[plys.size()-1]->addPoint(pnt_offset+1);
		plys[plys.size()-1]->addPoint(pnt_offset+2);
		plys[plys.size()-1]->addPoint(pnt_offset+3);
		plys[plys.size()-1]->addPoint(pnt_offset);
	}
}
#endif

} // end namespace FileIO

/**
 * \file GridAdapter.h
 * 24/03/2010 KR Initial implementation
 *
 */

#ifndef GRIDADAPTER_H
#define GRIDADAPTER_H

// ** INCLUDES **
#include "msh_mesh.h"

class vtkImageData; // For conversion from Image to QuadMesh
class vtkUnstructuredGrid; // For conversion vom vtk to ogs mesh

namespace MeshLib
{
class CFEMesh;
class CNode;
}

/**
 * \brief Adapter class to convert FEM Mesh to a representation more suited for visualisation purposes
 */
class GridAdapter
{
public:
	/// An element structure consisting of a number of nodes and a MshElemType
	typedef struct
	{
		MshElemType::type type;
		size_t material;
		std::vector<size_t> nodes;
	} Element;

	/// Constructor using a FEM-Mesh Object as source
	GridAdapter(const MeshLib::CFEMesh* mesh);

	/// Constructor using a MSH-file as source
	GridAdapter(const std::string &filename);

	/// Copy Constructor
	GridAdapter(const GridAdapter* grid = NULL);

	~GridAdapter();

	/// Adds a node to the grid
	void addNode(GEOLIB::Point* node) { _nodes->push_back(node); };

	/// Adds an element to the grid
	void addElement(Element* element) { _elems->push_back(element); };

	/// Returns the total number of unique material IDs.
	size_t getNumberOfMaterials() const;

	/// Returns the vector of nodes.
	const std::vector<GEOLIB::Point*>* getNodes() const { return _nodes; }

	/// Returns the vector of elements.
	const std::vector<Element*>* getElements() const { return _elems; }

	/// Return a vector of elements for one material group only.
	const std::vector<Element*>* getElements(size_t matID) const;

	/// Returns the grid as a CFEMesh for use in OGS-FEM
	const MeshLib::CFEMesh* getCFEMesh() const;

	/// Returns the grid as a CFEMesh for use in OGS-FEM
	const MeshLib::CFEMesh* getCFEMesh();

	/// Returns the name of the mesh.
	std::string getName() const { return _name; }

	/// Sets the element vector of the grid
	void setElements(std::vector<Element*> *elements) { _elems=elements; };

	/// Sets the name for the mesh.
	void setName(const std::string &name) { _name = name; }

	/// Sets the node vector of the grid
	void setNodeVector(std::vector<GEOLIB::Point*> *nodes) { _nodes=nodes; };

private:
	/// Converts an FEM Mesh to a list of nodes and elements.
	int convertCFEMesh(const MeshLib::CFEMesh* mesh);

	/// Reads a MSH file into a list of nodes and elements.
	int readMeshFromFile(const std::string &filename);

	/// Converts a string to a MshElemType
	MshElemType::type getElementType(const std::string &t) const;

	/// Converts a GridAdapter into an CFEMesh.
	const MeshLib::CFEMesh* toCFEMesh() const;

	std::string _name;
	std::vector<GEOLIB::Point*>* _nodes;
	std::vector<Element*>* _elems;
	const MeshLib::CFEMesh* _mesh;
};

#endif // GRIDADAPTER_H

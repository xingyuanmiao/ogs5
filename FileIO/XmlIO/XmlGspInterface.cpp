/**
 * \file XmlGspInterface.cpp
 * 2011/11/23 KR as derived class from XMLInterface
 */

#include "XmlGspInterface.h"

#include "XmlGmlInterface.h"
#include "XmlStnInterface.h"
#include "XmlCndInterface.h"

#include <QFileInfo>
#include <QFile>
#include <QtXml/QDomDocument>

namespace FileIO
{

XmlGspInterface::XmlGspInterface(ProjectData* project, const std::string &schemaFile)
: XMLInterface(project, schemaFile)
{
}

int XmlGspInterface::readFile(const QString &fileName)
{
	QFile* file = new QFile(fileName);
	QFileInfo fi(fileName);
	QString path = (fi.path().length() > 3) ? QString(fi.path() + "/") : fi.path();

	QFileInfo si(QString::fromStdString(_schemaName));
	QString schemaPath(si.absolutePath() + "/");

	if (!file->open(QIODevice::ReadOnly | QIODevice::Text))
	{
		std::cout << "XmlGspInterface::readFile() - Can't open xml-file " <<
		fileName.toStdString() << "." << "\n";
		delete file;
		return 0;
	}
	if (!checkHash(fileName))
	{
		delete file;
		return 0;
	}

	QDomDocument doc("OGS-PROJECT-DOM");
	doc.setContent(file);
	QDomElement docElement = doc.documentElement(); //OpenGeoSysProject
	if (docElement.nodeName().compare("OpenGeoSysProject"))
	{
		std::cout << "XmlGspInterface::readFile() - Unexpected XML root." << "\n";
		delete file;
		return 0;
	}

	QDomNodeList fileList = docElement.childNodes();

	for(int i = 0; i < fileList.count(); i++)
	{
		const QString file_node(fileList.at(i).nodeName());
		if (file_node.compare("geo") == 0)
		{
			XmlGmlInterface gml(_project, schemaPath.toStdString() + "OpenGeoSysGLI.xsd");
			const QDomNodeList childList = fileList.at(i).childNodes();
			for(int j = 0; j < childList.count(); j++)
			{
				const QDomNode child_node (childList.at(j));
				if (child_node.nodeName().compare("file") == 0) 
				{
					std::cout << "path: " << path.toStdString() << "#" << "\n";
					std::cout << "file name: " << (child_node.toElement().text()).toStdString() << "#" << "\n";
					gml.readFile(QString(path + child_node.toElement().text()));
				}
			}
		}
		else if (file_node.compare("stn") == 0)
		{
			XmlStnInterface stn(_project, schemaPath.toStdString() + "OpenGeoSysSTN.xsd");
			const QDomNodeList childList = fileList.at(i).childNodes();
			for(int j = 0; j < childList.count(); j++)
				if (childList.at(j).nodeName().compare("file") == 0)
					stn.readFile(QString(path + childList.at(j).toElement().text()));
		}
		else if (file_node.compare("msh") == 0)
		{
			const std::string msh_name = path.toStdString() +
			                       fileList.at(i).toElement().text().toStdString();
			FileIO::OGSMeshIO meshIO;
			MeshLib::CFEMesh* msh = meshIO.loadMeshFromFile(msh_name);
			QFileInfo fi(QString::fromStdString(msh_name));
			std::string name = fi.fileName().toStdString();
			_project->addMesh(msh, name);
			//GridAdapter msh(fileList.at(i).toElement().text().toStdString());
			// TODO gridadapter to mesh-models
		}
	}

	return 1;
}

int XmlGspInterface::writeToFile(std::string filename)
{
	_filename = filename;
	return FileIO::Writer::writeToFile(filename);
}

int XmlGspInterface::write(std::ostream& stream)
{
	GEOLIB::GEOObjects* geoObjects = _project->getGEOObjects();
	QFileInfo fi(QString::fromStdString(_filename));
	std::string path((fi.absolutePath()).toStdString() + "/");

	stream << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n"; // xml definition
	stream << "<?xml-stylesheet type=\"text/xsl\" href=\"OpenGeoSysProject.xsl\"?>\n\n"; // stylefile definition

	QDomDocument doc("OGS-PROJECT-DOM");
	QDomElement root = doc.createElement("OpenGeoSysProject");
	root.setAttribute( "xmlns:ogs", "http://www.opengeosys.org" );
	root.setAttribute( "xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance" );
	root.setAttribute( "xsi:noNamespaceSchemaLocation", "http://www.opengeosys.org/images/xsd/OpenGeoSysProject.xsd" );

	doc.appendChild(root);

	// GLI
	std::vector<std::string> geoNames;
	geoObjects->getGeometryNames(geoNames);
	for (std::vector<std::string>::const_iterator it(geoNames.begin()); it != geoNames.end();
	     ++it)
	{
		// write GLI file
		XmlGmlInterface gml(_project, path + "OpenGeoSysGLI.xsd");
		std::string name(*it);
		gml.setNameForExport(name);
		if (gml.writeToFile(std::string(path + name + ".gml")))
		{
			// write entry in project file
			QDomElement geoTag = doc.createElement("geo");
			root.appendChild(geoTag);
			QDomElement fileNameTag = doc.createElement("file");
			geoTag.appendChild(fileNameTag);
			QDomText fileNameText = doc.createTextNode(QString::fromStdString(name + ".gml"));
			fileNameTag.appendChild(fileNameText);
		}
	}

	// MSH
	const std::map<std::string, MeshLib::CFEMesh*> msh_vec = _project->getMeshObjects();
	for (std::map<std::string, MeshLib::CFEMesh*>::const_iterator it(msh_vec.begin());
	     it != msh_vec.end(); ++it)
	{
		// write mesh file
		std::string fileName(path + it->first);
		FileIO::OGSMeshIO meshIO;
		meshIO.setMesh(it->second);
		meshIO.writeToFile(fileName);

		// write entry in project file
		QDomElement mshTag = doc.createElement("msh");
		root.appendChild(mshTag);
		QDomElement fileNameTag = doc.createElement("file");
		mshTag.appendChild(fileNameTag);
		QDomText fileNameText = doc.createTextNode(QString::fromStdString(it->first));
		fileNameTag.appendChild(fileNameText);
	}

	// STN
	std::vector<std::string> stnNames;
	geoObjects->getStationVectorNames(stnNames);
	for (std::vector<std::string>::const_iterator it(stnNames.begin()); it != stnNames.end();
	     ++it)
	{
		// write STN file
		XmlStnInterface stn(_project, path + "OpenGeoSysSTN.xsd");
		std::string name(*it);
		stn.setNameForExport(name);

		if (stn.writeToFile(path + name + ".stn"))
		{
			// write entry in project file
			QDomElement geoTag = doc.createElement("stn");
			root.appendChild(geoTag);
			QDomElement fileNameTag = doc.createElement("file");
			geoTag.appendChild(fileNameTag);
			QDomText fileNameText = doc.createTextNode(QString::fromStdString(name + ".stn"));
			fileNameTag.appendChild(fileNameText);
		}
		else
			std::cout << "XmlGspInterface::writeFile() -  Error writing file: " << name << "\n";
	}

	std::string xml = doc.toString().toStdString();
	stream << xml;
	return 1;
}

}

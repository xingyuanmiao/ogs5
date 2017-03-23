#!/usr/bin/env bash

LIBS_LOCATION="$SOURCE_LOCATION/../Libs"

mkdir -vp $LIBS_LOCATION
cd $LIBS_LOCATION

QMAKE_LOCATION=`which qmake`

QT_VERSION="qt-everywhere-opensource-src-4.7.4"
VTK_VERSION="vtk-5.8.0"
SHAPELIB_VERSION="shapelib-1.3.0b2"
LIBGEOTIFF_VERSION="libgeotiff-1.3.0"
INSTANTCLIENT_VERSION="instantclient_11_2"

## Windows specific
if [ "$OSTYPE" == 'msys' ]; then
	if [ -z "$QMAKE_LOCATION" ]; then
		# Install Qt
		if [ ! -d qt ]; then
			# Download and extract
			download_file http://get.qt.nokia.com/qt/source/$QT_VERSION.zip ./$QT_VERSION.zip
			7za x $QT_VERSION.zip
			mv $QT_VERSION/ qt/
			rm $QT_VERSION.zip

		elif [ -f qt/bin/qmake.exe -a -f qt/bin/QtGui4.dll -a -f qt/bin/QtGuid4.dll ]; then
			# Already installed
			QT_FOUND=true
		fi

		if [ $QT_FOUND ]; then
			echo "Qt already installed in $LIBS_LOCATION/qt"
		else
			# Compile
			QT_CONFIGURATION="-debug-and-release"

			# Get instantclient
			QT_SQL_ARGS=""
			if [ $QT_SQL ]; then
				if [ ! -d instantclient ]; then
					if [ "$ARCHITECTURE" == "x64" ]; then
						download_file http://dl.dropbox.com/u/5581063/instantclient_11_2_x64.zip ./instantclient_11_2_x64.zip 015bd1b163571988cacf70e7d6185cb5
						7za x instantclient_11_2_x64.zip
						mv instantclient_11_2/ instantclient/
						rm instantclient_11_2_x64.zip
					fi
				fi
				QT_SQL_ARGS="-qt-sql-oci -I %cd%\..\instantclient\sdk\include -L %cd%\..\instantclient\sdk\lib\msvc"
			fi

			cd qt

			echo " \
			\"$WIN_DEVENV_PATH\\..\\..\\VC\\vcvarsall.bat\" $WIN_ARCHITECTURE &&\
			echo y | configure -opensource -no-accessibility -no-dsp -no-vcproj -no-phonon -no-webkit -no-scripttools -nomake demos -nomake examples $QT_CONFIGURATION $QT_SQL_ARGS &&\
			jom && nmake clean &&\
			exit\
			" > build.bat

			$COMSPEC \/k build.bat
			QT_WAS_BUILT=true
		fi

		export PATH=$PATH:$LIBS_LOCATION/qt/bin

	else
		echo "Qt already installed in $QMAKE_LOCATION"
	fi


	# Install VTK
	cd $LIBS_LOCATION
	if [ ! -d vtk ]; then
		# Download, extract, rename
		download_file http://www.vtk.org/files/release/5.8/$VTK_VERSION.tar.gz ./$VTK_VERSION.tar.gz
		tar -xf $VTK_VERSION.tar.gz
		rm $VTK_VERSION.tar.gz
	# Check for existing installation
	elif [ -f vtk/build/bin/Release/QVTK.lib -a -f vtk/build/bin/Release/vtkRendering.lib ]; then
		if [ $LIB_DEBUG ]; then
			if [ -f vtk/build/bin/Debug/QVTK.lib -a -f vtk/build/bin/Debug/vtkRendering.lib ]; then
				VTK_FOUND=true
			fi
		else
			VTK_FOUND=true
		fi
	fi

	if [ $VTK_FOUND ]; then
		echo "VTK already installed in $LIBS_LOCATION/vtk"
	else
		# Compile
		cd vtk
		mkdir -vp build
		cd build
		cmake .. -DBUILD_TESTING=OFF -DVTK_USE_QT=ON -G "$CMAKE_GENERATOR"
		cmake ..
		#$COMSPEC \/c "echo %PATH%"
		$COMSPEC \/c "devenv VTK.sln /Build Release"
		$COMSPEC \/c "devenv VTK.sln /Build Release /Project QVTK"
		$COMSPEC \/c "devenv VTK.sln /Build Debug"
		$COMSPEC \/c "devenv VTK.sln /Build Debug /Project QVTK"
	fi

	# Install shapelib
	cd $LIBS_LOCATION
	if [ ! -d shapelib ]; then
		# Download, extract
		download_file http://download.osgeo.org/shapelib/$SHAPELIB_VERSION.tar.gz ./$SHAPELIB_VERSION.tar.gz
		tar -xf $SHAPELIB_VERSION.tar.gz
		mv $SHAPELIB_VERSION/ shapelib/
		rm -rf $SHAPELIB_VERSION.tar.gz
	elif [ -f shapelib/shapelib.lib ]; then
		SHAPELIB_FOUND=true
	fi

	if [ $SHAPELIB_FOUND ]; then
		echo "Shapelib already installed in $LIBS_LOCATION/shapelib"
	else
		# Compile
		cd shapelib

		echo " \
		\"$WIN_DEVENV_PATH\\..\\..\\VC\\vcvarsall.bat\" $WIN_ARCHITECTURE &&\
		nmake /f makefile.vc &&\
		exit\
		" > build.bat

		$COMSPEC \/k build.bat
	fi

	# Install libgeotiff
	cd $LIBS_LOCATION
	if [ ! -d libgeotiff ]; then
		# Download, extract
		download_file http://download.osgeo.org/geotiff/libgeotiff/$LIBGEOTIFF_VERSION.tar.gz ./$LIBGEOTIFF_VERSION.tar.gz
		tar -xf $LIBGEOTIFF_VERSION.tar.gz
		mv $LIBGEOTIFF_VERSION/ libgeotiff/
		rm -rf $LIBGEOTIFF_VERSION.tar.gz
	elif [ -f libgeotiff/geotiff.lib ]; then
		LIBGEOTIFF_FOUND=true
	fi

	if [ $LIBGEOTIFF_FOUND ]; then
		echo "Libgeotiff already installed in $LIBS_LOCATION/libgeotiff"
	else
		# Compile
		cd libgeotiff

		# Download modified makefile
		if [ ! -f makefile_mod.vc ]; then
			download_file http://dl.dropbox.com/u/5581063/makefile_mod.vc ./makefile_mod.vc 14fb13a5bd04ffc298fee7825dc7679f
		fi

		echo " \
		\"$WIN_DEVENV_PATH\\..\\..\\VC\\vcvarsall.bat\" $WIN_ARCHITECTURE &&\
		nmake /f makefile_mod.vc all&&\
		exit\
		" > build.bat

		$COMSPEC \/k build.bat
	fi
fi

cd $SOURCE_LOCATION/scripts/setup
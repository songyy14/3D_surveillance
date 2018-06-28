#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <thread>
#include <chrono>
#include <mutex>

#include <osgViewer/Viewer>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Point>
#include <osg/Texture2D>
#include <osgDB/Readfile>
#include <osg/ShapeDrawable>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

using namespace std;
using namespace cv;
using namespace Eigen;


// Read camera matrix of one camera from file.
MatrixXd read_camera_matrix(string path);

// Calculate z coordinate given the plane equation and x, y coordinate.
double getZ(Vector4d plane, double x, double y);

// Calculate the point where a ray traveling through camera center and the given pixel intersects
// the given plane.
Vector3d intersect_plane(MatrixXd& camera_matrix, int u, int v, Vector4d& plane);

// Back project the pixel in image plane to ground plane in world coordinate system.
void back_projection(MatrixXd camera_matrix, int u, int v, int w, int h, osg::ref_ptr<osg::Vec3Array>& out);

// Read 3D point data from 3D-reconstruction result file.
void read_points(string file_points, osg::ref_ptr<osg::Vec3Array>& vertices, osg::ref_ptr<osg::Vec3Array>& colors);

// Create a drawable of point cloud from data file.
osg::ref_ptr<osg::Geometry> create_point_cloud(string file_points, float point_size);

// Create a geode of the ground plane.
osg::ref_ptr<osg::Geode> create_ground();

// Create a drawable of a sphere.
osg::ref_ptr<osg::ShapeDrawable> create_sphere(osg::Vec3 center, osg::Vec4 color);

#endif
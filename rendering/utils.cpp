#include <Windows.h>

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

#include "utils.h"

using namespace std;
using namespace cv;
using namespace Eigen;


Vector4d plane_ground(0.455776744955624, -6.20233437717713, 0.973225449420875, 1);
Vector4d plane_ground_our(0, 0, 1, 0);


MatrixXd read_camera_matrix(string path) {
	fstream file(path, fstream::in);
	MatrixXd camera_matrix(3, 4);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			file >> camera_matrix(i, j);

	return camera_matrix;
}


double getZ(Vector4d plane, double x, double y) {
	double z = -(plane(0) * x + plane(1) * y + plane(3)) / plane(2);
	return z;
}


Vector3d intersect_plane(MatrixXd& camera_matrix, int u, int v, Vector4d& plane) {
	MatrixXd A(4, 4);
	Vector4d b;

	// Preparation.
	A.block(0, 0, 3, 3) = camera_matrix.block(0, 0, 3, 3);
	A.block(3, 0, 1, 3) = plane.transpose().block(0, 0, 1, 3);
	A(0, 3) = -u; A(1, 3) = -v; A(2, 3) = -1; A(3, 3) = 0;

	b.block(0, 0, 3, 1) = (-1) * camera_matrix.block(0, 3, 3, 1);
	b(3) = (-1) * plane(3);

	// Get result.
	Vector4d tmp = A.inverse() * b;
	Vector3d result = tmp.block(0, 0, 3, 1);
	return result;
}


void back_projection(MatrixXd camera_matrix, int u, int v, int w, int h, osg::ref_ptr<osg::Vec3Array>& out) {
	
	MatrixXd transfer(2, 3);
	transfer << -0.103014405284553, 0.0899138313008130, 0.131139405691057,
		-0.0211405569105691, -0.00981263983739840, 0.627341938211382;

	// First, back-project the two bottom corners to the ground plane in our world coordinate system.
	Vector3d _bottom_left = intersect_plane(camera_matrix, u, v + h, plane_ground_our);
	Vector3d _bottom_right = intersect_plane(camera_matrix, u + w, v + h, plane_ground_our);

	// Then transfer their coordinates from our world coordinate system
	// to point-cloud coordinate system.
	_bottom_left(2) = 1;
	_bottom_right(2) = 1;

	Vector3d bottom_left, bottom_right;
	bottom_left.block(0, 0, 2, 1) = transfer * _bottom_left;
	bottom_right.block(0, 0, 2, 1) = transfer * _bottom_right;
	bottom_left(2) = getZ(plane_ground, bottom_left(0), bottom_left(1));
	bottom_right(2) = getZ(plane_ground, bottom_right(0), bottom_right(1));

	// Second, we wish to obtain a rectangle which is orthogonal to the ground plane,
	// and maintain the original aspect ratio of the bounding box;
	Vector3d normal_ground = plane_ground.block(0, 0, 3, 1);
	double k = h * (bottom_left - bottom_right).norm() / (w * normal_ground.norm());

	Vector3d top_left, top_right;
	top_left = bottom_left + k * normal_ground;
	top_right = bottom_right + k * normal_ground;

	// Output.
	out->push_back(osg::Vec3(top_left(0), top_left(1), top_left(2)));
	out->push_back(osg::Vec3(top_right(0), top_right(1), top_right(2)));
	out->push_back(osg::Vec3(bottom_right(0), bottom_right(1), bottom_right(2)));
	out->push_back(osg::Vec3(bottom_left(0), bottom_left(1), bottom_left(2)));
}


void read_points(string file_points, osg::ref_ptr<osg::Vec3Array>& vertices, osg::ref_ptr<osg::Vec3Array>& colors) {

	fstream file(file_points, fstream::in);
	string line;
	int num_points;
	int left, right;
	float data[6];

	// Determine number of points.
	getline(file, line);
	getline(file, line);
	getline(file, line);
	left = line.find(':');
	num_points = stoi(line.substr(left + 2, line.find(',') - left + 2));

	// Then each line corresponds to one point.
	for (int i = 0; i < num_points; i++) {
		getline(file, line);

		left = line.find(' ');
		// Read 3D-coordinate and RGB value.
		for (int j = 0; j < 6; j++) {
			right = line.find(' ', left + 1);
			data[j] = stof(line.substr(left + 1, right - left - 1));
			left = right;

			//cout << data[j] << " ";
		}
		//cout << endl;
		vertices->push_back(osg::Vec3(data[0], data[1], data[2]));
		colors->push_back(osg::Vec3(data[3] / 255.0, data[4] / 255.0, data[5] / 255.0));
	}
}


osg::ref_ptr<osg::Geometry> create_point_cloud(string file_points, float point_size) {

	osg::ref_ptr<osg::Geometry> point_cloud = new osg::Geometry;
	osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
	osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;
	osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;

	read_points(file_points, vertices, colors);
	normals->push_back(osg::Vec3(0, -1, 0));

	// Set vertices infomation.
	point_cloud->setVertexArray(vertices);
	point_cloud->setColorArray(colors);
	point_cloud->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	point_cloud->setNormalArray(normals);
	point_cloud->setNormalBinding(osg::Geometry::BIND_OVERALL);

	// Set how to arrange the vertices.
	point_cloud->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, vertices->size()));

	// Set the point size.
	point_cloud->getOrCreateStateSet()->setAttribute(new osg::Point(point_size), osg::StateAttribute::ON);

	// The point cloud is static, and OpenSceneGraph renders dynamic drawables first then static drawables to reduce the chance of data racing.
	point_cloud->setDataVariance(osg::Object::STATIC);

	return point_cloud.release();

}


osg::ref_ptr<osg::Geode> create_ground() {

	double x1, y1, x2, y2;
	x1 = -10; y1 = -10; x2 = 10, y2 = 10;
	double z1, z2, z3, z4;
	z1 = getZ(plane_ground, x1, y1);
	z2 = getZ(plane_ground, x2, y1);
	z3 = getZ(plane_ground, x2, y2);
	z4 = getZ(plane_ground, x1, y2);

	osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
	vertices->push_back(osg::Vec3(x1, y1, z1));
	vertices->push_back(osg::Vec3(x2, y1, z2));
	vertices->push_back(osg::Vec3(x2, y2, z3));
	vertices->push_back(osg::Vec3(x1, y2, z4));

	osg::ref_ptr<osg::Vec3Array> colors = new osg::Vec3Array;
	colors->push_back(osg::Vec3(0.95, 0.95, 0.95));

	osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
	normals->push_back(osg::Vec3(0, -1, 0));

	osg::ref_ptr<osg::Geometry> ground = new osg::Geometry;
	ground->setVertexArray(vertices);
	ground->setColorArray(colors);
	ground->setColorBinding(osg::Geometry::BIND_OVERALL);
	ground->setNormalArray(normals);
	ground->setNormalBinding(osg::Geometry::BIND_OVERALL);

	ground->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

	osg::ref_ptr<osg::Geode> result = new osg::Geode;
	result->addDrawable(ground);

	return result.release();
}


osg::ref_ptr<osg::ShapeDrawable> create_sphere(osg::Vec3 center, osg::Vec4 color) {

	float radius = 0.005;
	osg::ref_ptr<osg::ShapeDrawable> sphere = new osg::ShapeDrawable;
	sphere->setShape(new osg::Sphere(center, radius));
	sphere->setColor(color);

	return sphere.release();
}
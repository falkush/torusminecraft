#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>


static uint8_t* buffer = 0;
static uint8_t* buffer2 = 0;
static bool* blocks1 = 0;
static bool* blocks2 = 0;
cudaError_t cudaStatus;

__device__ double gauss(double x, double sigma)
{
	return exp((-1.0 / 2.0) * x * x * (1.0 / sigma) * (1.0 / sigma));
}

__device__ int collcolor2(bool* blocks, int collblock, int tmpx, int tmpy, bool rem, double coord1, double coord2, double d1, double d2, int face, bool other, int currx, int curry, int currz)
{
	int i;
	double r, g, b;
	int contid;
	double fact;

	if (rem && tmpx == 1920 / 2 && tmpy == 1080 / 2) blocks[collblock] = false;

	for (i = 0; i < 3; i++) collblock = (60493 * collblock + 11) % 115249;

	if (other)
	{
		collblock = (60493 * collblock + 11) % 115249;

		if (currz == 0)
		{
			if (collblock % 3 == 0)
			{
				r = 255.0;
				g = 0.0;
				b = 255.0;
			}
			else if (collblock % 3 == 1)
			{
				r = 0.0;
				g = 255.0;
				b = 255.0;
			}
			else
			{
				r = 255.0;
				g = 255.0;
				b = 255.0;
			}
		}
		else
		{
			if (collblock % 1001 == 0)
			{
				r = 128.0;
				g = 0.0;
				b = 255.0;
			}
			else if (collblock % 5 == 0)
			{
				r = 128.0;
				g = 128.0;
				b = 128.0;
			}
			else if (collblock % 5 == 1)
			{
				r = 200.0;
				g = 200.0;
				b = 200.0;
			}
			else if (collblock % 5 == 2)
			{
				r = 135.0;
				g = 42.0;
				b = 1.0;
			}
			else if (collblock % 5 == 3)
			{
				r = 64.0;
				g = 64.0;
				b = 64.0;
			}
			else
			{
				r = 83.0;
				g = 41.0;
				b = 0.0;
			}
		}
	}
	else
	{
		if (currz == 0)
		{
			if (curry < 19)
			{
				r = 0.0;
				g = 0.0;
				b = 255.0;
			}
			else if (curry < 20)
			{
				r = 0.0;
				g = 255.0;
				b = 255.0;
			}
			else if (curry < 21)
			{
				r = 254.0;
				g = 214.0;
				b = 131.0;
			}
			else if (curry < 22)
			{
				r = 0.0;
				g = 214.0;
				b = 0.0;
			}
			else if (curry < 27)
			{
				r = 0.0;
				g = 64.0;
				b = 0.0;
			}
			else if (curry < 28)
			{
				r = 0.0;
				g = 214.0;
				b = 0.0;
			}
			else if (curry < 29)
			{
				r = 254.0;
				g = 214.0;
				b = 131.0;
			}
			else if (curry < 30)
			{
				r = 0.0;
				g = 255.0;
				b = 255.0;
			}
		}
		else
		{
			if (collblock % 1001 == 0)
			{
				r = 128.0;
				g = 0.0;
				b = 255.0;
			}
			else if (collblock % 5 == 0)
			{
				r = 128.0;
				g = 128.0;
				b = 128.0;
			}
			else if (collblock % 5 == 1)
			{
				r = 200.0;
				g = 200.0;
				b = 200.0;
			}
			else if (collblock % 5 == 2)
			{
				r = 135.0;
				g = 42.0;
				b = 1.0;
			}
			else if (collblock % 5 == 3)
			{
				r = 64.0;
				g = 64.0;
				b = 64.0;
			}
			else
			{
				r = 83.0;
				g = 41.0;
				b = 0.0;
			}
		}
	}

	contid = floor(10.0 * coord1 / d1) + 10 * floor(10.0 * coord2 / d2);
	contid += collblock;

	for (i = 0; i < 10; i++) contid = (60493 * contid + 11) % 115249;
	if (contid < 0) contid += 115249;
	fact = 0.3 * contid / 115248.0;

	r = (1.0 - fact) * r;
	g = (1.0 - fact) * g;
	b = (1.0 - fact) * b;


	fact = face * 0.1;

	r = (1.0 - fact) * r;
	g = (1.0 - fact) * g;
	b = (1.0 - fact) * b;

	return (int)r + 256 * (int)g + 256 * 256 * (int)b;
}

__device__ int raymarch1(bool* blocks1, double rayon, double alpha, double dx, double dy, double dz, int currx, int curry, int currz, double vecn0, double vecn1, double vl, double pos0, double pos1, int nbx, int nby, double kappa, int tmpx, int tmpy, bool rem, bool other, bool add)
{
	long count = 0;
	double contx, conty, contz;
	double coordx, coordy, coordz;
	int height = floor((rayon - alpha) / dz);

	double nextcoordz = alpha + currz * dz;
	int nbxnby = nbx * nby;

	int signx = (-2) * signbit(vecn0) + 1;
	int signy = (-2) * signbit(vecn1) + 1;

	vecn0 *= signx;
	vecn1 *= signy;

	int currblock = currx + nbx * curry + nbxnby * currz;

	if (blocks1[currblock])
	{
		coordx = pos0;
		coordy = pos1;

		coordx = fmod(coordx, dx);
		coordy = fmod(coordy, dy);

		if (coordx < 0) coordx += dx;
		if (coordy < 0) coordy += dy;

		return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 0, other, currx, curry, currz);
	}


	double t0dx = dx * vl / vecn0;
	double t0dy = dy * vl / vecn1;

	double sigx = fmod(pos0, dx);
	double sigy = fmod(pos1, dy);

	int face;

	if (sigx < 0) sigx += dx;
	if (sigy < 0) sigy += dy;

	if (signx == 1) contx = t0dx - sigx * vl / vecn0;
	else contx = sigx * vl / vecn0;

	if (signy == 1) conty = t0dy - sigy * vl / vecn1;
	else conty = sigy * vl / vecn1;



	if (currz < height)
	{
		nextcoordz += dz;
		contz = kappa - sqrt(rayon * rayon - nextcoordz * nextcoordz);

		count = 0;
		while (count < 100)
		{
			count++;
			if (contz < contx)
			{
				if (contz < conty)
				{
					currz++;
					currblock += nbxnby;
					if (blocks1[currblock])
					{
						coordx = pos0 + contz * vecn0 * signx / vl;
						coordy = pos1 + contz * vecn1 * signy / vl;

						coordx = fmod(coordx, dx);
						coordy = fmod(coordy, dy);

						if (coordx < 0) coordx += dx;
						if (coordy < 0) coordy += dy;

						return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 0, other, currx, curry, currz);
					}
					else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

					if (currz == height)
					{
						contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz);
						break;
					}

					nextcoordz += dz;
					contz = kappa - sqrt(rayon * rayon - nextcoordz * nextcoordz);

				}
				else
				{
					curry += signy;
					curry %= nby;
					if (curry < 0) curry += nby;

					currblock = currx + nbx * curry + nbxnby * currz;
					if (blocks1[currblock])
					{
						coordx = pos0 + conty * vecn0 * signx / vl;
						coordx = fmod(coordx, dx);
						if (coordx < 0) coordx += dx;

						coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
						coordz -= alpha;
						coordz = fmod(coordz, dz);

						if (signy == 1) face = 4;
						else face = 2;

						return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
					}
					else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

					conty += t0dy;
				}
			}
			else
			{
				if (contx < conty)
				{
					currx += signx;
					currx %= nbx;
					if (currx < 0) currx += nbx;

					currblock = currx + nbx * curry + nbxnby * currz;
					if (blocks1[currblock])
					{
						coordy = pos1 + contx * vecn1 * signy / vl;
						coordy = fmod(coordy, dy);
						if (coordy < 0) coordy += dy;

						coordz = sqrt(rayon * rayon - (contx - kappa) * (contx - kappa));
						coordz -= alpha;
						coordz = fmod(coordz, dz);

						if (signx == 1) face = 3;
						else face = 1;

						return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordy, coordz, dy, dz, face, other, currx, curry, currz);
					}
					else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

					contx += t0dx;
				}
				else
				{
					curry += signy;
					curry %= nby;
					if (curry < 0) curry += nby;

					currblock = currx + nbx * curry + nbxnby * currz;
					if (blocks1[currblock])
					{
						coordx = pos0 + conty * vecn0 * signx / vl;
						coordx = fmod(coordx, dx);
						if (coordx < 0) coordx += dx;

						coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
						coordz -= alpha;
						coordz = fmod(coordz, dz);

						if (signy == 1) face = 4;
						else face = 2;

						return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
					}
					else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

					conty += t0dy;
				}
			}
		}
	}
	else contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz);

	count = 0;
	while (count < 100)
	{
		count++;
		if (contz < contx)
		{
			if (contz < conty)
			{
				if (currz == 0) return -1;
				currz--;

				currblock -= nbxnby;

				if (blocks1[currblock])
				{
					coordx = pos0 + contz * vecn0 * signx / vl;
					coordy = pos1 + contz * vecn1 * signy / vl;

					coordx = fmod(coordx, dx);
					coordy = fmod(coordy, dy);

					if (coordx < 0) coordx += dx;
					if (coordy < 0) coordy += dy;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 5, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				nextcoordz -= dz;
				contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz);

			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
		else
		{
			if (contx < conty)
			{
				currx += signx;
				currx %= nbx;
				if (currx < 0) currx += nbx;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordy = pos1 + contx * vecn1 * signy / vl;
					coordy = fmod(coordy, dy);
					if (coordy < 0) coordy += dy;

					coordz = sqrt(rayon * rayon - (contx - kappa) * (contx - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signx == 1) face = 3;
					else face = 1;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordy, coordz, dy, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				contx += t0dx;
			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;
				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
	}
	return 0;
}

__device__ int raymarch2(bool* blocks1, double rayon, double alpha, double dx, double dy, double dz, int currx, int curry, int currz, double vecn0, double vecn1, double vl, double pos0, double pos1, int nbx, int nby, double kappa, int tmpx, int tmpy, bool rem, bool other, bool add)
{
	long count;
	double contx, conty, contz;
	double coordx, coordy, coordz;
	int face;

	double nextcoordz = alpha + currz * dz;
	int nbxnby = nbx * nby;

	int signx = (-2) * signbit(vecn0) + 1;
	int signy = (-2) * signbit(vecn1) + 1;

	vecn0 *= signx;
	vecn1 *= signy;

	int currblock = currx + nbx * curry + nbxnby * currz;

	double t0dx = dx * vl / vecn0;
	double t0dy = dy * vl / vecn1;

	double sigx = fmod(pos0, dx);
	double sigy = fmod(pos1, dy);

	if (sigx < 0) sigx += dx;
	if (sigy < 0) sigy += dy;

	if (signx == 1) contx = t0dx - sigx * vl / vecn0;
	else contx = sigx * vl / vecn0;

	if (signy == 1) conty = t0dy - sigy * vl / vecn1;
	else conty = sigy * vl / vecn1;

	contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz);

	count = 0;
	while (count < 100)
	{
		count++;
		if (contz < contx)
		{
			if (contz < conty)
			{
				if (currz == 0) return -1;
				currz--;
				currblock -= nbxnby;

				if (blocks1[currblock])
				{
					coordx = pos0 + contz * vecn0 * signx / vl;
					coordy = pos1 + contz * vecn1 * signy / vl;

					coordx = fmod(coordx, dx);
					coordy = fmod(coordy, dy);

					if (coordx < 0) coordx += dx;
					if (coordy < 0) coordy += dy;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 5, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				nextcoordz -= dz;
				contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz);

			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;
				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
		else
		{
			if (contx < conty)
			{
				currx += signx;
				currx %= nbx;
				if (currx < 0) currx += nbx;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordy = pos1 + contx * vecn1 * signy / vl;
					coordy = fmod(coordy, dy);
					if (coordy < 0) coordy += dy;

					coordz = sqrt(rayon * rayon - (contx - kappa) * (contx - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signx == 1) face = 3;
					else face = 1;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordy, coordz, dy, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				contx += t0dx;
			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;
				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
	}

	return 0;
}

__device__ int raymarch3(bool* blocks1, bool* blocks2, double rayon, double alpha, double dx, double dy, double dz, int currx, int curry, int currz, double vecn0, double vecn1, double vl, double pos0, double pos1, int nbx, int nby, int nbz, double kappa, int tmpx, int tmpy, bool rem, bool other, bool add)
{
	long count;
	double contx, conty, contz;
	double coordx, coordy, coordz;
	int face;
	double nextcoordz = alpha + currz * dz + dz;
	contz = kappa - sqrt(rayon * rayon - nextcoordz * nextcoordz);
	int nbxnby = nbx * nby;

	double sub = 2.0 * sqrt(rayon * rayon - (alpha + dz * nbz) * (alpha + dz * nbz));

	int signx = (-2) * signbit(vecn0) + 1;
	int signy = (-2) * signbit(vecn1) + 1;

	vecn0 *= signx;
	vecn1 *= signy;

	int currblock = currx + nbx * curry + nbxnby * currz;

	if (blocks1[currblock])
	{
		coordx = pos0;
		coordy = pos1;

		coordx = fmod(coordx, dx);
		coordy = fmod(coordy, dy);

		if (coordx < 0) coordx += dx;
		if (coordy < 0) coordy += dy;

		return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 0, other, currx, curry, currz);
	}

	double t0dx = dx * vl / vecn0;
	double t0dy = dy * vl / vecn1;

	double sigx = fmod(pos0, dx);
	double sigy = fmod(pos1, dy);

	if (sigx < 0) sigx += dx;
	if (sigy < 0) sigy += dy;

	if (signx == 1) contx = t0dx - sigx * vl / vecn0;
	else contx = sigx * vl / vecn0;

	if (signy == 1) conty = t0dy - sigy * vl / vecn1;
	else conty = sigy * vl / vecn1;



	count = 0;
	while (count < 100)
	{
		count++;
		if (contz < contx)
		{
			if (contz < conty)
			{
				if (currz == nbz - 1) break;
				currz++;
				currblock += nbxnby;

				if (blocks1[currblock])
				{
					coordx = pos0 + contz * vecn0 * signx / vl;
					coordy = pos1 + contz * vecn1 * signy / vl;

					coordx = fmod(coordx, dx);
					coordy = fmod(coordy, dy);

					if (coordx < 0) coordx += dx;
					if (coordy < 0) coordy += dy;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 0, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				nextcoordz += dz;
				contz = kappa - sqrt(rayon * rayon - nextcoordz * nextcoordz);

			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
		else
		{
			if (contx < conty)
			{
				currx += signx;
				currx %= nbx;
				if (currx < 0) currx += nbx;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordy = pos1 + contx * vecn1 * signy / vl;
					coordy = fmod(coordy, dy);
					if (coordy < 0) coordy += dy;

					coordz = sqrt(rayon * rayon - (contx - kappa) * (contx - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signx == 1) face = 3;
					else face = 1;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordy, coordz, dy, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				contx += t0dx;
			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks1[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty - kappa) * (conty - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks1, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks1[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
	}

	other = !other;

	if (blocks2[currblock])
	{
		coordx = pos0 + contz * vecn0 * signx / vl;
		coordy = pos1 + contz * vecn1 * signy / vl;

		coordx = fmod(coordx, dx);
		coordy = fmod(coordy, dy);

		if (coordx < 0) coordx += dx;
		if (coordy < 0) coordy += dy;

		return collcolor2(blocks2, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 5, other, currx, curry, currz);
	}
	else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks2[currblock] = true; return 0; }

	nextcoordz -= dz;
	contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz) - sub;

	count = 0;
	while (count < 100)
	{
		count++;
		if (contz < contx)
		{
			if (contz < conty)
			{
				if (currz == 0) return -1;
				currz--;

				currblock -= nbxnby;

				if (blocks2[currblock])
				{
					coordx = pos0 + contz * vecn0 * signx / vl;
					coordy = pos1 + contz * vecn1 * signy / vl;

					coordx = fmod(coordx, dx);
					coordy = fmod(coordy, dy);

					if (coordx < 0) coordx += dx;
					if (coordy < 0) coordy += dy;

					return collcolor2(blocks2, currblock, tmpx, tmpy, rem, coordx, coordy, dx, dy, 5, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks2[currblock] = true; return 0; }


				nextcoordz -= dz;
				contz = kappa + sqrt(rayon * rayon - nextcoordz * nextcoordz) - sub;

			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks2[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty + sub - kappa) * (conty + sub - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks2, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks2[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
		else
		{
			if (contx < conty)
			{
				currx += signx;
				currx %= nbx;
				if (currx < 0) currx += nbx;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks2[currblock])
				{
					coordy = pos1 + contx * vecn1 * signy / vl;
					coordy = fmod(coordy, dy);
					if (coordy < 0) coordy += dy;

					coordz = sqrt(rayon * rayon - (contx + sub - kappa) * (contx + sub - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signx == 1) face = 3;
					else face = 1;

					return collcolor2(blocks2, currblock, tmpx, tmpy, rem, coordy, coordz, dy, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks2[currblock] = true; return 0; }

				contx += t0dx;
			}
			else
			{
				curry += signy;
				curry %= nby;
				if (curry < 0) curry += nby;

				currblock = currx + nbx * curry + nbxnby * currz;

				if (blocks2[currblock])
				{
					coordx = pos0 + conty * vecn0 * signx / vl;
					coordx = fmod(coordx, dx);
					if (coordx < 0) coordx += dx;

					coordz = sqrt(rayon * rayon - (conty + sub - kappa) * (conty + sub - kappa));
					coordz -= alpha;
					coordz = fmod(coordz, dz);

					if (signy == 1) face = 4;
					else face = 2;

					return collcolor2(blocks2, currblock, tmpx, tmpy, rem, coordx, coordz, dx, dz, face, other, currx, curry, currz);
				}
				else if (add && tmpx == 1920 / 2 && tmpy == 1080 / 2) { blocks2[currblock] = true; return 0; }

				conty += t0dy;
			}
		}
	}
	return 0;
}

__device__ void tormat(double phi, double theta, double* mat)
{
	mat[0] = cos(theta) * sin(phi);
	mat[3] = cos(theta) * cos(phi);
	mat[6] = sin(theta);

	mat[2] = cos(phi);
	mat[5] = -sin(phi);
	mat[8] = 0;

	mat[1] = -sin(theta) * sin(phi);
	mat[4] = -sin(theta) * cos(phi);
	mat[7] = cos(theta);
}

__device__ double matdet(double* m)
{
	return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

__device__ void matinv(double* m, double* res)
{
	res[0] = m[4] * m[8] - m[5] * m[7];
	res[1] = m[2] * m[7] - m[1] * m[8];
	res[2] = m[1] * m[5] - m[2] * m[4];
	res[3] = m[5] * m[6] - m[3] * m[8];
	res[4] = m[0] * m[8] - m[2] * m[6];
	res[6] = m[3] * m[7] - m[4] * m[6];
	res[5] = m[2] * m[3] - m[0] * m[5];
	res[7] = m[1] * m[6] - m[0] * m[7];
	res[8] = m[0] * m[4] - m[1] * m[3];
}

__device__ void matmult(double* m1, double* m2, double* res)
{
	res[0] = m1[0] * m2[0] + m1[1] * m2[3] + m1[2] * m2[6];
	res[1] = m1[0] * m2[1] + m1[1] * m2[4] + m1[2] * m2[7];
	res[2] = m1[0] * m2[2] + m1[1] * m2[5] + m1[2] * m2[8];
	res[3] = m1[3] * m2[0] + m1[4] * m2[3] + m1[5] * m2[6];
	res[4] = m1[3] * m2[1] + m1[4] * m2[4] + m1[5] * m2[7];
	res[5] = m1[3] * m2[2] + m1[4] * m2[5] + m1[5] * m2[8];
	res[6] = m1[6] * m2[0] + m1[7] * m2[3] + m1[8] * m2[6];
	res[7] = m1[6] * m2[1] + m1[7] * m2[4] + m1[8] * m2[7];
	res[8] = m1[6] * m2[2] + m1[7] * m2[5] + m1[8] * m2[8];
}

__device__ void matact(double* m, double vecn0, double vecn1, double vecn2, double* nvecn)
{
	nvecn[0] = m[0] * vecn0 + m[1] * vecn1 + m[2] * vecn2;
	nvecn[1] = m[3] * vecn0 + m[4] * vecn1 + m[5] * vecn2;
	nvecn[2] = m[6] * vecn0 + m[7] * vecn1 + m[8] * vecn2;
}

__device__ void matflip(double* m, double* res)
{
	res[0] = m[6];
	res[1] = m[7];
	res[2] = m[8];
	res[3] = m[3];
	res[4] = m[4];
	res[5] = m[5];
	res[6] = -m[0];
	res[7] = -m[1];
	res[8] = -m[2];
}

__device__ void matflip2(double* m, double* res)
{
	res[0] = m[2];
	res[1] = m[1];
	res[2] = -m[0];
	res[3] = m[5];
	res[4] = m[4];
	res[5] = -m[3];
	res[6] = m[8];
	res[7] = m[7];
	res[8] = -m[6];
}

__device__ double solvequartic(double a0, double b0, double c0, double d0, double e0)
{


	double tmp;
	double tmin = 65536.0;
	double sint, s;
	double r1, qds, rootint;

	double b = b0 / a0;
	double c = c0 / a0;
	double d = d0 / a0;
	double e = e0 / a0;

	double c2 = c * c;
	double bd = b * d;
	double c3 = c2 * c;
	double bcd = bd * c;
	double b2 = b * b;
	double b2e = b2 * e;
	double d2 = d * d;
	double ce = c * e;
	double bc = b * c;
	double b3 = b2 * b;
	double mbd4 = (-0.25) * b;

	double t0 = c2 - 3.0 * bd + 12.0 * e;
	double t1 = 2.0 * c3 - 9.0 * bcd + 27.0 * b2e + 27.0 * d2 - 72.0 * ce;
	double p = (8.0 * c - 3.0 * b2) / 8.0;
	double q = (b3 - 4.0 * bc + 8.0 * d) / 8.0;

	double disc = t1 * t1 - 4.0 * t0 * t0 * t0;

	if (disc < 0)
	{
		double st0 = sqrt(t0);
		double phi = (acos(t1 / (2.0 * t0 * st0))) / 3.0;
		sint = (-2.0 / 3.0) * p + (2.0 / 3.0) * st0 * cos(phi);
	}
	else
	{
		double bigq = cbrt((t1 + sqrt(disc)) * 0.5);
		sint = (-2.0 / 3.0) * p + (1.0 / 3.0) * (bigq + t0 / bigq);
	}

	s = sqrt(sint) * 0.5;
	//if (abs(s) < 0.000001) return 65536.0;

	rootint = (sint + 2.0 * p) * (-1.0);
	qds = q / s;

	r1 = rootint + qds;

	if (r1 > 0)
	{
		r1 = 0.5 * sqrt(r1);
		tmp = mbd4 - s;

		if (tmp + r1 > 0.0000001 && tmp + r1 < tmin) tmin = tmp + r1;
		if (tmp - r1 > 0.0000001 && tmp - r1 < tmin) tmin = tmp - r1;
	}

	r1 = rootint - qds;

	if (r1 > 0)
	{
		r1 = 0.5 * sqrt(r1);
		tmp = mbd4 + s;

		if (tmp + r1 > 0.0000001 && tmp + r1 < tmin) tmin = tmp + r1;
		if (tmp - r1 > 0.0000001 && tmp - r1 < tmin) tmin = tmp - r1;
	}

	//double sol2, sol3, sol4;
	//sol2 = tmin * tmin;
	//sol3 = sol2 * tmin;
	//sol4 = sol3 * tmin;

	//if (abs(a0 * sol4 + b0 * sol3 + c0 * sol2 + d0 * tmin + e0) < 0.01 && tmin<10 && tmin>0.000001) return tmin;
	//else return 65536.0;
	return tmin;
}

__device__ double toruscoll(double a, double b, double c, double d, double e, double f, double m, double n)
{
	double t4, t3, t2, t1, t0;

	double a2 = a * a;
	double b2 = b * b;
	double c2 = c * c;
	double d2 = d * d;
	double e2 = e * e;
	double f2 = f * f;

	double ab = a * b;
	double cd = c * d;
	double ef = e * f;

	double sum1 = a2 + c2 + e2;
	double sum2 = ab + cd;
	double sum3 = sum2 + ef;
	double sum4 = m + n;
	double sum5 = b2 + d2 + f2;
	double sum6 = m - n;
	double sum7 = ab + ef;

	t0 = sum5 * sum5 + sum6 * sum6;
	t0 += (-2.0) * (sum5 * sum4 - 2.0 * f2 * n);

	t1 = (b2 + d2 + f2) * sum3;
	t1 -= sum3 * sum4;
	t1 += 2.0 * ef * n;
	t1 *= 4.0;

	t2 = d * (d * (sum1 + 2.0 * c2) + 4.0 * c * sum7) + b * (b * (sum1 + 2.0 * a2) + 4.0 * a * ef) + f2 * (sum1 + 2.0 * e2);
	t2 -= sum1 * sum4;
	t2 += 2.0 * e2 * n;
	t2 *= 2.0;

	t3 = 4.0 * sum1 * sum3;

	t4 = sum1 * sum1;

	return solvequartic(t4, t3, t2, t1, t0);
}

__global__ void bufferinit(uint8_t* buffer, uint8_t* buffer2)
{
	buffer[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 3] = 255;
	buffer2[4 * (blockIdx.x * blockDim.x + threadIdx.x) + 3] = 255;
}

__global__ void setblocks(bool* blocks1, bool* blocks2)
{
	for (int i = 0; i < 30 * 30 * 30; i++)
	{
		blocks1[i] = true;
		blocks2[i] = true;
	}

}

__global__ void setblocksrand(bool* blocks1, bool* blocks2)
{
	int i;
	int rand = 1;

	for (i = 0; i < 30 * 30 * 30; i++)
	{
		rand = (60493 * rand + 11) % 115249;
		if (rand % 100 == 0) blocks1[i] = true;
		else blocks1[i] = false;
		rand = (60493 * rand + 11) % 115249;
		if (rand % 100 == 0) blocks2[i] = true;
		else blocks2[i] = false;
	}

}

__global__ void addKernel(uint8_t* buffer, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, bool inside, double alpha, double beta, double bigr, double r, bool other, bool* blocks1, bool* blocks2, double dx, double dy, double dz, int currx, int curry, int currz, int nbx, int nby, int nbz, bool rem, int frame, bool add, double alphaef, double alphaef2, double alphaef3, double alphaef4)
{


	int i;
	double vecn0, vecn1, vecn2;

	double geoang;
	double xyvec;

	double inv[9]{};
	double nvecn[3]{};
	double npos[3]{};
	double vl;

	double torcoll;
	double tor0, tor1, tor2;
	double theta, phi;

	double exitalpha;

	double rayon;
	double kappa;
	double exit;

	int collrgb;
	double skyr, skyg, skyb;

	double kesum;

	double si = 0;
	int ix, iy;
	double mx, my;

	int rand = 1;
	int new0, new1, new2;

	double u, v;
	uint8_t uv;

	int tmp2, tmpx2, tmpy2;
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;
	int tmpx = tmp % 1920;
	int tmpy = (tmp - tmpx) / 1920;

	int newrpos, newgpos, newbpos;

	double vecl;

	double mat1[9]{};

	uint8_t retr;
	uint8_t retg;
	uint8_t retb;

	double tmp0ef, tmp1ef, tmp2ef, tmp3ef;

	double betaef = 0.75 * 0.7;
	double epsilonef = 1.0 / 30.0;

	double gamma0ef, gamma1ef, gamma2ef;

	vecn0 = vec0 + tmpx * addy0 + tmpy * addz0;
	vecn1 = vec1 + tmpx * addy1 + tmpy * addz1;
	vecn2 = vec2 + tmpx * addy2 + tmpy * addz2;

	vecn0 += alphaef * 30 * sin(0.01 * frame + 0.007 * tmpx) * addy0;
	vecn1 += alphaef * 30 * sin(0.01 * frame + 0.007 * tmpx) * addy1;
	vecn2 += alphaef * 30 * sin(0.01 * frame + 0.007 * tmpx) * addy2;


	vecn0 += alphaef * 20 * sin(0.01 * frame + 0.006 * tmpy) * addz0;
	vecn1 += alphaef * 20 * sin(0.01 * frame + 0.006 * tmpy) * addz1;
	vecn2 += alphaef * 20 * sin(0.01 * frame + 0.006 * tmpy) * addz2;



	vecl = sqrt(vecn0 * vecn0 + vecn1 * vecn1 + vecn2 * vecn2);

	vecn0 /= vecl;
	vecn1 /= vecl;
	vecn2 /= vecl;

	alphaef2 *= 0.95;

	if (inside)
	{
		vl = sqrt(1.0 - vecn2 * vecn2);
		geoang = atan(vecn2 / vl);
		rayon = pos2 / cos(geoang);
		kappa = sin(geoang) * rayon;
		exitalpha = sqrt(rayon * rayon - alpha * alpha);
		kesum = kappa + exitalpha;

		if (kappa > 0)
		{
			if (beta < rayon)
			{
				if (other) collrgb = raymarch3(blocks2, blocks1, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, nbz, kappa, tmpx, tmpy, rem, other, add);
				else collrgb = raymarch3(blocks1, blocks2, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, nbz, kappa, tmpx, tmpy, rem, other, add);

				if (collrgb != -1)
				{
					retr = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retg = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retb = collrgb % 256;
					goto end;
				}

				exit = sqrt(rayon * rayon - beta * beta);
				kesum -= 2.0 * exit;
				other = !other;
			}
			else
			{
				if (other) collrgb = raymarch1(blocks2, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, kappa, tmpx, tmpy, rem, true, add);
				else collrgb = raymarch1(blocks1, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, kappa, tmpx, tmpy, rem, false, add);

				if (collrgb != -1)
				{
					retr = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retg = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retb = collrgb % 256;
					goto end;
				}
			}
		}
		else
		{
			if (other) collrgb = raymarch2(blocks2, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, kappa, tmpx, tmpy, rem, other, add);
			else collrgb = raymarch2(blocks1, rayon, alpha, dx, dy, dz, currx, curry, currz, vecn0, vecn1, vl, pos0, pos1, nbx, nby, kappa, tmpx, tmpy, rem, other, add);

			if (collrgb != -1)
			{
				retr = collrgb % 256;
				collrgb -= collrgb % 256;
				collrgb /= 256;
				retg = collrgb % 256;
				collrgb -= collrgb % 256;
				collrgb /= 256;
				retb = collrgb % 256;
				goto end;
			}
		}



		pos0 += kesum * vecn0 / vl;
		pos1 += kesum * vecn1 / vl;

		vecn0 = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * vecn0 / vl;
		vecn1 = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * vecn1 / vl;
		vecn2 = -exitalpha / rayon;

		pos0 /= bigr;
		pos1 /= r;

		npos[0] = sin(pos0) * (bigr + r * cos(pos1));
		npos[1] = cos(pos0) * (bigr + r * cos(pos1));
		npos[2] = r * sin(pos1);

		tormat(pos0, pos1, mat1);
		matflip2(mat1, inv);
		matact(inv, vecn0, vecn1, vecn2, nvecn);

		vecn0 = nvecn[0];
		vecn1 = nvecn[1];
		vecn2 = nvecn[2];

		pos0 = npos[0] + vecn0 * 0.001;
		pos1 = npos[1] + vecn1 * 0.001;
		pos2 = npos[2] + vecn2 * 0.001;
	}

	torcoll = toruscoll(vecn0, pos0, vecn1, pos1, vecn2, pos2, r * r, bigr * bigr);

	for (i = 0; i < 10; i++) {
		if (torcoll != 65536)
		{
			tor0 = pos0 + torcoll * vecn0;
			tor1 = pos1 + torcoll * vecn1;
			tor2 = pos2 + torcoll * vecn2;
			xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

			if (abs(tor2 / r) > 1 || abs(tor1 / xyvec) > 1)
			{
				retr = 0;
				retg = 0;
				retb = 0;
				goto end;
			}

			theta = asin(tor2 / r);
			if (xyvec < bigr) theta = M_PI - theta;
			if (theta < 0) theta += 2.0 * M_PI;

			phi = acos(tor1 / xyvec);
			if (tor0 < 0) phi *= -1.0;
			if (phi < 0) phi += 2.0 * M_PI;

			tormat(phi, theta, mat1);
			matinv(mat1, inv);
			matflip(inv, mat1);
			matact(mat1, vecn0, vecn1, vecn2, nvecn);

			npos[0] = phi * (bigr);
			npos[1] = theta * r;

			vl = sqrt(1.0 - nvecn[2] * nvecn[2]);
			geoang = atan(nvecn[2] / vl);
			rayon = alpha / cos(geoang);
			kappa = sin(geoang) * rayon;

			currx = floor(npos[0] * nbx / (2.0 * M_PI * bigr));
			curry = floor(npos[1] * nby / (2.0 * M_PI * r));

			if (beta < rayon)
			{
				if (other) collrgb = raymarch3(blocks2, blocks1, rayon, alpha, dx, dy, dz, currx, curry, 0, nvecn[0], nvecn[1], vl, npos[0], npos[1], nbx, nby, nbz, kappa, tmpx, tmpy, rem, other, add);
				else collrgb = raymarch3(blocks1, blocks2, rayon, alpha, dx, dy, dz, currx, curry, 0, nvecn[0], nvecn[1], vl, npos[0], npos[1], nbx, nby, nbz, kappa, tmpx, tmpy, rem, other, add);

				if (collrgb != -1)
				{
					retr = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retg = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retb = collrgb % 256;
					goto end;
				}

				exitalpha = sqrt(rayon * rayon - alpha * alpha);
				exit = sqrt(rayon * rayon - beta * beta);

				npos[0] += (kappa - 2.0 * exit + exitalpha) * nvecn[0] / vl;
				npos[1] += (kappa - 2.0 * exit + exitalpha) * nvecn[1] / vl;

				nvecn[0] = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * nvecn[0] / vl;
				nvecn[1] = sqrt(1.0 - (exitalpha * exitalpha) / (rayon * rayon)) * nvecn[1] / vl;
				nvecn[2] = -exitalpha / rayon;

				other = !other;
			}
			else
			{

				if (other) collrgb = raymarch1(blocks2, rayon, alpha, dx, dy, dz, currx, curry, 0, nvecn[0], nvecn[1], vl, npos[0], npos[1], nbx, nby, kappa, tmpx, tmpy, rem, other, add);
				else collrgb = raymarch1(blocks1, rayon, alpha, dx, dy, dz, currx, curry, 0, nvecn[0], nvecn[1], vl, npos[0], npos[1], nbx, nby, kappa, tmpx, tmpy, rem, other, add);

				if (collrgb != -1)
				{
					retr = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retg = collrgb % 256;
					collrgb -= collrgb % 256;
					collrgb /= 256;
					retb = collrgb % 256;
					goto end;
				}

				npos[0] += 2.0 * kappa * nvecn[0] / vl;
				npos[1] += 2.0 * kappa * nvecn[1] / vl;

				nvecn[2] *= -1.0;
			}



			npos[0] /= bigr;
			npos[1] /= r;


			pos0 = sin(npos[0]) * (bigr + r * cos(npos[1]));
			pos1 = cos(npos[0]) * (bigr + r * cos(npos[1]));
			pos2 = r * sin(npos[1]);

			tormat(npos[0], npos[1], mat1);
			matflip2(mat1, inv);
			matact(inv, nvecn[0], nvecn[1], nvecn[2], nvecn);

			vecn0 = nvecn[0];
			vecn1 = nvecn[1];
			vecn2 = nvecn[2];

			pos0 += vecn0 * 0.00001;
			pos1 += vecn1 * 0.00001;
			pos2 += vecn2 * 0.00001;

			torcoll = toruscoll(vecn0, pos0, vecn1, pos1, vecn2, pos2, r * r, bigr * bigr);
		}
	}


	if (torcoll != 65536)
	{
		retr = 0;
		retg = 0;
		retb = 0;
		goto end;
	}

	if (other)
	{

		u = 0.5 + atan2(vecn1, vecn0) / (2.0 * M_PI);
		v = 0.5 + asin(vecn2) / M_PI;

		ix = u * 1000.0;
		iy = v * 1000.0;
		mx = ix + 0.5;
		my = iy + 0.5;
		mx /= 1000;
		my /= 1000;
		rand = ix + 1000 * iy;
		rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249;
		if (rand < 0) rand += 115249;

		retr = 0;
		retg = 0;
		retb = 0;

		if (rand % 5 == 0)
		{
			tmp3ef = sqrt((u - mx) * (u - mx) + (v - my) * (v - my));
			if (tmp3ef > 1.0 / 2000.0) si = 0;
			else si = exp(-10.0 * rand / 115249.0) * 255.0 * exp(-4000.0 * (tmp3ef));
			retr = si;
			retg = si;
			retb = si;
		}

		ix = u * 100.0;
		iy = v * 100.0;
		mx = ix + 0.5;
		my = iy + 0.5;
		mx /= 100;
		my /= 100;
		rand = ix + 100 * iy;
		rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249; rand = (1664525 * rand + 11) % 115249;
		if (rand < 0) rand += 115249;

		if (rand % 17 == 0)
		{
			tmp3ef = sqrt((u - mx) * (u - mx) + (v - my) * (v - my));
			if (tmp3ef > 1.0 / 200.0) si = 0;
			else si = 255.0 * (exp(-700.0 * tmp3ef));

			if (rand % 7 == 0)
			{
				retr = si;
				retg = 0;
				retb = 0;
			}
			else if (rand % 7 == 1)
			{
				retr = 0;
				retg = si;
				retb = 0;
			}
			else if (rand % 7 == 2)
			{
				retr = 0;
				retg = 0;
				retb = si;
			}
			else if (rand % 7 == 3)
			{
				retr = si;
				retg = si;
				retb = 0;
			}
			else if (rand % 7 == 4)
			{
				retr = si;
				retg = 0;
				retb = si;
			}
			else if (rand % 7 == 5)
			{
				retr = 0;
				retg = si;
				retb = si;
			}
			else if (rand % 7 == 6)
			{
				retr = si;
				retg = si;
				retb = si;
			}
		}




		retr = exp(-15.0 * v) * 255.0 + (1 - exp(-15.0 * v)) * retr;
		retg = exp(-15.0 * v) * 255.0 + (1 - exp(-15.0 * v)) * retg;
		retb = exp(-15.0 * v) * 255.0 + (1 - exp(-15.0 * v)) * retb;

		retr = exp(-15.0 * (1 - v)) * 0 + (1 - exp(-15.0 * (1 - v))) * retr;
		retg = exp(-15.0 * (1 - v)) * 255.0 + (1 - exp(-15.0 * (1 - v))) * retg;
		retb = exp(-15.0 * (1 - v)) * 255.0 + (1 - exp(-15.0 * (1 - v))) * retb;

	}
	else
	{
		skyr = 96.0;
		skyg = 149.0;
		skyb = 217.0;

		v = ((0.5 + asin(vecn2) / M_PI));

		retr = v * 255.0 + (1 - v) * skyr;
		retg = v * 255.0 + (1 - v) * skyg;
		retb = v * 255.0 + (1 - v) * skyb;
	}

end:


	tmp0ef = retr;
	tmp1ef = retg;
	tmp2ef = retb;

	tmp0ef /= 255.0;
	tmp1ef /= 255.0;
	tmp2ef /= 255.0;

	gamma0ef = sin(frame * epsilonef + 2.0 * M_PI * tmp1ef + 2.0 * M_PI * tmp2ef);
	gamma1ef = sin(frame * epsilonef * 0.71 + 2.0 * M_PI * tmp2ef + 2.0 * M_PI * tmp0ef);
	gamma2ef = sin(frame * epsilonef * 0.47 + 2.0 * M_PI * tmp0ef + 2.0 * M_PI * tmp1ef);

	tmp0ef = pow(tmp0ef, 1.0 + alphaef * betaef * gamma0ef);
	tmp1ef = pow(tmp1ef, 1.0 + alphaef * betaef * gamma1ef);
	tmp2ef = pow(tmp2ef, 1.0 + alphaef * betaef * gamma2ef);



	tmp0ef *= 255;
	tmp1ef *= 255;
	tmp2ef *= 255;


	tmp0ef *= (1.0 - alphaef2);
	tmp1ef *= (1.0 - alphaef2);
	tmp2ef *= (1.0 - alphaef2);

	tmp0ef += alphaef2 * buffer[4 * tmp];
	tmp1ef += alphaef2 * buffer[4 * tmp + 1];
	tmp2ef += alphaef2 * buffer[4 * tmp + 2];



	new0 = (int)round(tmpx + alphaef3 * 30.0 * sin(frame / 30.0));
	new1 = (int)round(tmpx + alphaef3 * 30.0 * sin(frame / 30.0 + 2.0 * M_PI / 3.0));
	new2 = (int)round(tmpx + alphaef3 * 30.0 * sin(frame / 30.0 + 4.0 * M_PI / 3.0));

	if (new0 < 0)
	{
		new0 += 1920;
		tmp0ef = 0;
	}
	else if (new0 >= 1920)
	{
		new0 -= 1920;
		tmp0ef = 0;
	}

	if (new1 < 0)
	{
		new1 += 1920;
		tmp1ef = 0;
	}
	else if (new1 >= 1920)
	{
		new1 -= 1920;
		tmp1ef = 0;
	}

	if (new2 < 0)
	{
		new2 += 1920;
		tmp2ef = 0;
	}
	else if (new2 >= 1920)
	{
		new2 -= 1920;
		tmp2ef = 0;
	}

	newrpos = 4 * (new0 + 1920 * tmpy);
	newgpos = 4 * (new1 + 1920 * tmpy) + 1;
	newbpos = 4 * (new2 + 1920 * tmpy) + 2;

	if (tmp0ef < 256 && 0 <= tmp0ef && tmp1ef < 256 && 0 <= tmp1ef && tmp2ef < 256 && 0 <= tmp2ef) {

		buffer[newrpos] = tmp0ef;
		buffer[newgpos] = tmp1ef;
		buffer[newbpos] = tmp2ef;
	}
	else
	{
		buffer[newrpos] = 0;
		buffer[newgpos] = 0;
		buffer[newbpos] = 0;
	}
}

__global__ void blur(uint8_t* buffer, uint8_t* buffer2)
{
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;
	int tmpx = tmp % 1920;
	int tmpy = (tmp - tmpx) / 1920;

	double w0 = 1.0 / sqrt(M_PI);
	double w1 = 1.0 / (M_E * sqrt(M_PI));
	double w2 = 1.0 / sqrt(M_PI * M_E);

	//double tw0 = w0 + w1 + w1 + w2; //corner
	//double tw1 = w0 + w1 + w1 + w1 + w2 + w2; //side
	double tw2 = w0 + w1 + w1 + w1 + w1 + w2 + w2 + w2 + w2; //inside

	if (tmpy != 0 && tmpx != 0 && tmpy != 1080 - 1 && tmpx != 1920 - 1) {
		buffer2[4 * tmp] = (buffer[4 * tmp] * w0 + w1 * (buffer[4 * ((tmpx + 1) + 1920 * tmpy)] + buffer[4 * ((tmpx - 1) + 1920 * tmpy)] + buffer[4 * ((tmpx)+1920 * (tmpy + 1))] + buffer[4 * ((tmpx)+1920 * (tmpy - 1))]) + w2 * (buffer[4 * ((tmpx + 1) + 1920 * (tmpy + 1))] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy + 1))] + buffer[4 * ((tmpx + 1) + 1920 * (tmpy - 1))] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy - 1))])) / tw2;
		buffer2[4 * tmp + 1] = (buffer[4 * tmp + 1] * w0 + w1 * (buffer[4 * ((tmpx + 1) + 1920 * tmpy) + 1] + buffer[4 * ((tmpx - 1) + 1920 * tmpy) + 1] + buffer[4 * ((tmpx)+1920 * (tmpy + 1)) + 1] + buffer[4 * ((tmpx)+1920 * (tmpy - 1)) + 1]) + w2 * (buffer[4 * ((tmpx + 1) + 1920 * (tmpy + 1)) + 1] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy + 1)) + 1] + buffer[4 * ((tmpx + 1) + 1920 * (tmpy - 1)) + 1] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy - 1)) + 1])) / tw2;
		buffer2[4 * tmp + 2] = (buffer[4 * tmp + 2] * w0 + w1 * (buffer[4 * ((tmpx + 1) + 1920 * tmpy) + 2] + buffer[4 * ((tmpx - 1) + 1920 * tmpy) + 2] + buffer[4 * ((tmpx)+1920 * (tmpy + 1)) + 2] + buffer[4 * ((tmpx)+1920 * (tmpy - 1)) + 2]) + w2 * (buffer[4 * ((tmpx + 1) + 1920 * (tmpy + 1)) + 2] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy + 1)) + 2] + buffer[4 * ((tmpx + 1) + 1920 * (tmpy - 1)) + 2] + buffer[4 * ((tmpx - 1) + 1920 * (tmpy - 1)) + 2])) / tw2;

	}
	else
	{
		buffer2[4 * tmp] = buffer[4 * tmp];
		buffer2[4 * tmp + 1] = buffer[4 * tmp + 1];
		buffer2[4 * tmp + 2] = buffer[4 * tmp + 2];
	}
}

void cudaInit()
{
	cudaSetDevice(0);
	cudaMalloc((void**)&buffer, 4 * 1920 * 1080 * sizeof(uint8_t));
	cudaMalloc((void**)&buffer2, 4 * 1920 * 1080 * sizeof(uint8_t));
	cudaMalloc((void**)&blocks1, 30 * 30 * 30);
	cudaMalloc((void**)&blocks2, 30 * 30 * 30);

	bufferinit << <(int)(1920 * 1080 / 600), 600 >> > (buffer, buffer2);
	cudaDeviceSynchronize();



	setblocks << <1, 1 >> > (blocks1, blocks2);
	cudaDeviceSynchronize();
}

void cudaExit()
{
	cudaFree(buffer);
	cudaFree(buffer2);
	cudaFree(blocks1);
	cudaFree(blocks2);
	cudaDeviceReset();
}

void cudathingy(uint8_t* pixels, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, bool inside, double alpha, double beta, double bigr, double r, bool other, double dx, double dy, double dz, int currx, int curry, int currz, int nbx, int nby, int nbz, bool* cpublocks1, bool* cpublocks2, bool rem, bool blockrand, bool reset, int frame, bool add, bool load, double alphaef, double alphaef2, double alphaef3, double alphaef4)
{
	int nbblurs = alphaef4 * 20.0;
	if (blockrand) setblocksrand << <1, 1 >> > (blocks1, blocks2);
	if (reset) setblocks << <1, 1 >> > (blocks1, blocks2);
	if (load)
	{
		cudaMemcpy(blocks1, cpublocks1, 30 * 30 * 30, cudaMemcpyHostToDevice);
		cudaMemcpy(blocks2, cpublocks2, 30 * 30 * 30, cudaMemcpyHostToDevice);
	}

	addKernel << <(int)(1920 * 1080 / 100), 100 >> > (buffer, pos0, pos1, pos2, vec0, vec1, vec2, addy0, addy1, addy2, addz0, addz1, addz2, inside, alpha, beta, bigr, r, other, blocks1, blocks2, dx, dy, dz, currx, curry, currz, nbx, nby, nbz, rem, frame, add, alphaef, alphaef2, alphaef3, alphaef4);
	cudaDeviceSynchronize();

	if (nbblurs % 2 == 0)
	{


		for (int i = 0; i < nbblurs / 2; i++)
		{
			blur << <(int)(1920 * 1080 / 100), 100 >> > (buffer, buffer2);
			cudaDeviceSynchronize();
			blur << <(int)(1920 * 1080 / 100), 100 >> > (buffer2, buffer);
			cudaDeviceSynchronize();
		}


		cudaMemcpy(pixels, buffer, 4 * 1920 * 1080 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	}
	else
	{
		blur << <(int)(1920 * 1080 / 100), 100 >> > (buffer, buffer2);
		cudaDeviceSynchronize();
		for (int i = 0; i < (nbblurs - 1) / 2; i++)
		{
			blur << <(int)(1920 * 1080 / 100), 100 >> > (buffer2, buffer);
			cudaDeviceSynchronize();
			blur << <(int)(1920 * 1080 / 100), 100 >> > (buffer, buffer2);
			cudaDeviceSynchronize();
		}


		cudaMemcpy(pixels, buffer2, 4 * 1920 * 1080 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	}

	if (rem || add)
	{
		cudaMemcpy(cpublocks1, blocks1, 30 * 30 * 30, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpublocks2, blocks2, 30 * 30 * 30, cudaMemcpyDeviceToHost);
	}
}
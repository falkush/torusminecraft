#define _USE_MATH_DEFINES

#include <SFML/Graphics.hpp>
#include "windows.h" 


void cudathingy(uint8_t* pixels, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, bool inside, double alpha, double beta, double bigr, double r, bool other, double dx, double dy, double dz, int currx, int curry, int currz, int nbx, int nby, int nbz, bool* blocks1, bool* blocks2, bool rem, bool blockrand, bool reset);
void cudaInit();
void cudaExit();

double toruscoll2(double a, double b, double c, double d, double e, double f, double m, double n);
double solvequartic2(double a0, double b0, double c0, double d0, double e0);
void tormat2(double phi, double theta, double* mat);
double matdet2(double* m);
void matinv2(double* m, double* res);
void matmult2(double* m1, double* m2, double* res);
void matact2(double* m, double vecn0, double vecn1, double vecn2, double* nvecn);
void matflipcpu(double* m, double* res);
void matflip2cpu(double* m, double* res);
void setblockscpu(bool* blocks1, bool* blocks2);
void setblocksrandcpu(bool* blocks1, bool* blocks2);

static bool blocks1[30 * 30 * 30]{};
static bool blocks2[30 * 30 * 30]{};

int main()
{
    ShowWindow(GetConsoleWindow(), SW_HIDE);
	//ShowWindow(GetConsoleWindow(), SW_SHOW);

	double r = 1.0;
	double bigr = 1.5;

	double roomsize = 10.0;
	double schecker = 1.0;
	double dist = 2.0;
	double sqsz = 0.01 / 4;
	double speed = 0.01;
	double alpha = 1.0;
	double beta = 5.0;


	double alpha2 = 1.0;
	double beta2 = beta - alpha;


	double outangle=0;

	bool other = false;

	double anglex, angley;
	double xl;
	double dotp;

	double vl, geoang;

	double tmppos0, tmppos1, tmppos2, tmpx00, tmpx01, tmpx02, tmpx10, tmpx11, tmpx12, tmpx20, tmpx21, tmpx22;

	bool inside = false;
	
	bool reset = false;


    int mousx, mousy, centralx, centraly;

    double pos0, pos1, pos2;
    double vec0, vec1, vec2;
    double addy0, addy1, addy2;
    double addz0, addz1, addz2;
    double x00, x01, x02;
    double x10, x11, x12;
    double x20, x21, x22;
    double multy = (1 - 1920) * sqsz / 2;
    double multz = (1080-1) * sqsz / 2;


    double newx00, newx01, newx02;
    double newx10, newx11, newx12;
    double newx20, newx21, newx22;

	double torcoll;
	double rayon;
	double newang;
	double guder;
	double distrem;



	double proj0, proj1;

	double tor0, tor1, tor2, xyvec, theta, phi;
	double mat1[9]{};
	double mat[9]{};
	double nvecn[3]{};
	double inv[9]{};
	double npos[3]{};
	
	int nbx = 30;
	int nby = 30;
	int nbz = 30;

	double dx = 2.0*M_PI*bigr / nbx;
	double dy = 2.0 * M_PI * r / nby;
	double dz = (beta - alpha) / nbz;
	int currx=0, curry=0, currz=0;

    bool focus = true;

	bool blockrand = false;
	bool tmpinside, tmpother;
	bool rem = false;


	setblockscpu(blocks1, blocks2);

   //sf::RenderWindow window(sf::VideoMode(1920, 1080, 32), "Torus Minecraft - Press ESC to stop", sf::Style::Titlebar | sf::Style::Close);
	sf::RenderWindow window(sf::VideoMode(1920, 1080, 32), "Torus Minecraft - Press ESC to stop", sf::Style::Fullscreen);
	sf::Texture texture;
    sf::Sprite sprite;
    sf::Uint8* pixels = new sf::Uint8[1920 * 1080 * 4];
    sf::Vector2i winpos;

    cudaInit();
    texture.create(1920, 1080);
    window.setMouseCursorVisible(false);

	x00 = 0.0; x01 = 0.0; x02 = -1.0;
	x10 = 0.0; x11 = 1.0; x12 = 0.0;
	x20 = 1.0; x21 = 0.0; x22 = 0.0;
	pos0 = 0.0; pos1 = 0.0; pos2 = 7.93;

    winpos = window.getPosition();
    SetCursorPos(winpos.x + 1920 / 2, winpos.y + 1080 / 2);
   
    while (window.isOpen())
    {
        //Sleep(1);
        sf::Event event;
		
		while (window.pollEvent(event))
		{
			if (focus && event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::J)
			{
				setblocksrandcpu(blocks1,blocks2);
				blockrand = true;
			}

			if (event.type == sf::Event::Closed)
				window.close();

			if (focus && event.type == sf::Event::MouseMoved)
			{
				POINT p;
				GetCursorPos(&p);
				winpos = window.getPosition();
				centralx = winpos.x + 1920 / 2;
				centraly = winpos.y + 1080 / 2;
				SetCursorPos(centralx, centraly);

				mousx = p.x - centralx;
				mousy = p.y - centraly;

				anglex = 0.002 * mousx;
				angley = 0.002 * mousy;

				if (anglex < 0) anglex *= -1;
				if (angley < 0) angley *= -1;


				if (mousx > 0)
				{
					newx00 = x00 * cos(anglex) + sin(anglex) * x10;
					newx10 = x10 * cos(anglex) - sin(anglex) * x00;
					x00 = newx00;
					x10 = newx10;

					newx01 = x01 * cos(anglex) + sin(anglex) * x11;
					newx11 = x11 * cos(anglex) - sin(anglex) * x01;
					x01 = newx01;
					x11 = newx11;

					newx02 = x02 * cos(anglex) + sin(anglex) * x12;
					newx12 = x12 * cos(anglex) - sin(anglex) * x02;
					x02 = newx02;
					x12 = newx12;
				}
				else if (mousx < 0)
				{
					newx00 = x00 * cos(anglex) - sin(anglex) * x10;
					newx10 = x10 * cos(anglex) + sin(anglex) * x00;
					x00 = newx00;
					x10 = newx10;

					newx01 = x01 * cos(anglex) - sin(anglex) * x11;
					newx11 = x11 * cos(anglex) + sin(anglex) * x01;
					x01 = newx01;
					x11 = newx11;

					newx02 = x02 * cos(anglex) - sin(anglex) * x12;
					newx12 = x12 * cos(anglex) + sin(anglex) * x02;
					x02 = newx02;
					x12 = newx12;
				}

				if (mousy < 0)
				{
					newx00 = x00 * cos(angley) + sin(angley) * x20;
					newx20 = x20 * cos(angley) - sin(angley) * x00;
					x00 = newx00;
					x20 = newx20;

					newx01 = x01 * cos(angley) + sin(angley) * x21;
					newx21 = x21 * cos(angley) - sin(angley) * x01;
					x01 = newx01;
					x21 = newx21;

					newx02 = x02 * cos(angley) + sin(angley) * x22;
					newx22 = x22 * cos(angley) - sin(angley) * x02;
					x02 = newx02;
					x22 = newx22;
				}
				else if (mousy > 0)
				{
					newx00 = x00 * cos(angley) - sin(angley) * x20;
					newx20 = x20 * cos(angley) + sin(angley) * x00;
					x00 = newx00;
					x20 = newx20;

					newx01 = x01 * cos(angley) - sin(angley) * x21;
					newx21 = x21 * cos(angley) + sin(angley) * x01;
					x01 = newx01;
					x21 = newx21;

					newx02 = x02 * cos(angley) - sin(angley) * x22;
					newx22 = x22 * cos(angley) + sin(angley) * x02;
					x02 = newx02;
					x22 = newx22;
				}


			}

			if (focus && event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) rem = true;
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
		{
			x00 = 0.0; x01 = 0.0; x02 = -1.0;
			x10 = 0.0; x11 = 1.0; x12 = 0.0;
			x20 = 1.0; x21 = 0.0; x22 = 0.0;
			pos0 = 0.0; pos1 = 0.0; pos2 = 7.93;
			inside = false;
			alpha2 = 1.0;
			bigr = 1.5;
			r = 1.0;
			alpha = 1.0;
			speed = 0.01;
			beta = 5.0;
			beta2 = beta - alpha;
			dx = 2.0 * M_PI * bigr / nbx;
			dy = 2.0 * M_PI * r / nby;
			dz = beta2 / nbz;
			other = false;

			setblockscpu(blocks1, blocks2);
			reset = true;
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z)) { bigr += 0.003; dx = 2.0 * M_PI * bigr / nbx; }
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::X)) { bigr -= 0.003; dx = 2.0 * M_PI * bigr / nbx; }
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) { r += 0.003;  alpha = alpha2 * r; beta = beta2 + alpha; dy=2.0 * M_PI * r / nby; dz = (beta - alpha) / nbz;}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::V)) { r -= 0.003;  alpha = alpha2 * r; beta = beta2 + alpha; dy = 2.0 * M_PI * r / nby; dz = (beta - alpha) / nbz;}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) { alpha2 += 0.003; alpha = alpha2 * r; beta = beta2 + alpha; dz = (beta - alpha) / nbz;}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::N)) { alpha2 -= 0.003;  alpha = alpha2 * r; beta = beta2 + alpha; dz = (beta - alpha) / nbz;}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::K)) speed += 0.0001;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::L)) if(speed-0.0001>0) speed -= 0.0001;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::G)) {beta2 += 0.03; beta += 0.03; dz = (beta - alpha) / nbz;}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::H)) {beta2 -= 0.03; beta -= 0.03; dz = (beta - alpha) / nbz;}
		
		/*
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            {
                focus = false;
                window.setMouseCursorVisible(true);
            }*/
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();


            if(sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                focus = true;
                window.setMouseCursorVisible(false);
            }

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            {
				tmpinside = inside;
				tmpother = other;

				tmppos0 = pos0;
				tmppos1 = pos1;
				tmppos2 = pos2;

				tmpx00 = x00;
				tmpx01 = x01;
				tmpx02 = x02;
				tmpx10 = x10;
				tmpx11 = x11;
				tmpx12 = x12;
				tmpx20 = x20;
				tmpx21 = x21;
				tmpx22 = x22;

				if (inside)
				{
						vl = sqrt(1.0 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));
						
						newang = atan(sinh(guder - speed));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						rayon = pos2 / cos(geoang);
						pos2 = rayon * cos(newang);

						if (pos2 > beta)
						{
							newang = acos(beta / rayon);
							distrem = speed - guder + asinh(tan(newang));
							
							other = !other;

							mat[0] = x00;
							mat[3] = x01;
							mat[6] = x02;

							mat[1] = -sin(geoang) * proj0;
							mat[4] = -sin(geoang) * proj1;
							mat[7] = cos(geoang);

							mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
							mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
							mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

							mat1[0] = cos(newang) * proj0;
							mat1[3] = cos(newang) * proj1;
							mat1[6] = sin(newang);

							mat1[1] = -sin(newang) * proj0;
							mat1[4] = -sin(newang) * proj1;
							mat1[7] = cos(newang);

							mat1[2] = mat[2];
							mat1[5] = mat[5];
							mat1[8] = mat[8];

							matinv2(mat, inv);
							matmult2(mat1, inv, mat);

							x00 = mat1[0];
							x01 = mat1[3];
							x02 = -mat1[6];

							matact2(mat, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = -nvecn[2];
							matact2(mat, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = -nvecn[2];

							pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
							pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
							
							////////////////////////////////////////
							
							vl = sqrt(1.0 - x02 * x02);
							geoang = atan(x02 / vl);

							guder = asinh(tan(geoang));

							newang = atan(sinh(guder - distrem));

							proj0 = x00 / vl;
							proj1 = x01 / vl;

							pos2 = rayon * cos(newang);

							mat[0] = x00;
							mat[3] = x01;
							mat[6] = x02;

							mat[1] = -sin(geoang) * proj0;
							mat[4] = -sin(geoang) * proj1;
							mat[7] = cos(geoang);

							mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
							mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
							mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

							mat1[0] = cos(newang) * proj0;
							mat1[3] = cos(newang) * proj1;
							mat1[6] = sin(newang);

							mat1[1] = -sin(newang) * proj0;
							mat1[4] = -sin(newang) * proj1;
							mat1[7] = cos(newang);

							mat1[2] = mat[2];
							mat1[5] = mat[5];
							mat1[8] = mat[8];

							matinv2(mat, inv);
							matmult2(mat1, inv, mat);

							x00 = mat1[0];
							x01 = mat1[3];
							x02 = mat1[6];

							matact2(mat, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
							matact2(mat, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

							pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
							pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
							
							////////////////////////////////////////
						}
						else
						{
							if (pos2 < alpha)
							{
								inside = false;
								newang = -acos(alpha / rayon);
								distrem = speed - guder + asinh(tan(newang));
							}


							mat[0] = x00;
							mat[3] = x01;
							mat[6] = x02;

							mat[1] = -sin(geoang) * proj0;
							mat[4] = -sin(geoang) * proj1;
							mat[7] = cos(geoang);

							mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
							mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
							mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

							mat1[0] = cos(newang) * proj0;
							mat1[3] = cos(newang) * proj1;
							mat1[6] = sin(newang);

							mat1[1] = -sin(newang) * proj0;
							mat1[4] = -sin(newang) * proj1;
							mat1[7] = cos(newang);

							mat1[2] = mat[2];
							mat1[5] = mat[5];
							mat1[8] = mat[8];

							matinv2(mat, inv);
							matmult2(mat1, inv, mat);

							x00 = mat1[0];
							x01 = mat1[3];
							x02 = mat1[6];

							matact2(mat, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
							matact2(mat, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

							pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
							pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

							if (!inside)
							{
								pos0 /= bigr;
								pos1 /= r;

								npos[0] = sin(pos0) * (bigr + r * cos(pos1));
								npos[1] = cos(pos0) * (bigr + r * cos(pos1));
								npos[2] = r * sin(pos1);

								tormat2(pos0, pos1, mat1);
								matflip2cpu(mat1, inv);
								matact2(inv, x00, x01, x02, nvecn);
								x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
								matact2(inv, x10, x11, x12, nvecn);
								x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
								matact2(inv, x20, x21, x22, nvecn);
								x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

								pos0 = npos[0];
								pos1 = npos[1];
								pos2 = npos[2];

								pos0 += x00 * distrem;
								pos1 += x01 * distrem;
								pos2 += x02 * distrem;
							}
						}
				}
				else
				{
					torcoll = toruscoll2(x00, pos0, x01, pos1, x02, pos2, r * r, bigr * bigr);
					if (torcoll == 65536 || torcoll > speed)
					{
						pos0 += x00 * speed;
						pos1 += x01 * speed;
						pos2 += x02 * speed;
					}
					else
					{
						tor0 = pos0 + torcoll * x00;
						tor1 = pos1 + torcoll * x01;
						tor2 = pos2 + torcoll * x02;
						xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

						theta = asin(tor2 / r);
						if (xyvec < bigr) theta = M_PI - theta;
						if (theta < 0) theta += 2.0 * M_PI;

						phi = acos(tor1 / xyvec);
						if (tor0 < 0) phi *= -1;
						if (phi < 0) phi += 2.0 * M_PI;

						tormat2(phi, theta, mat1);
						matinv2(mat1, inv);
						matflipcpu(inv, mat1);
						matact2(mat1, x00, x01, x02, nvecn);
						x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
						matact2(mat1, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat1, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 = phi * (bigr);
						pos1 = theta * r;
						pos2 = alpha;

						inside = true;

						distrem = speed - torcoll;

						vl = sqrt(1 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						rayon = pos2 / cos(geoang);
						pos2 = rayon * cos(newang);


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
					}
				}

				if (inside)
				{
					pos0 = fmod(pos0, 2.0 * M_PI * bigr);
					pos0 += signbit(pos0) * 2 * M_PI * bigr;

					pos1 = fmod(pos1, 2.0 * M_PI * r);
					pos1 += signbit(pos1) * 2.0 * M_PI * r;

					currx = floor(pos0 * nbx / (2.0 * M_PI * bigr));
					curry = floor(pos1 * nby / (2.0 * M_PI * r));
					currz = floor((pos2 - alpha) * nbz / (beta - alpha));

					if (other)
					{
						if (blocks2[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}
					else
					{
						if (blocks1[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}
					
				}

            }

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			{
				newx00 = x00;
				newx01 = x01;
				newx02 = x02;

				x00 = x10;
				x01 = x11;
				x02 = x12;

				x10 = newx00;
				x11 = newx01;
				x12 = newx02;

				tmpinside = inside;
				tmpother = other;

				tmppos0 = pos0;
				tmppos1 = pos1;
				tmppos2 = pos2;

				tmpx00 = x00;
				tmpx01 = x01;
				tmpx02 = x02;
				tmpx10 = x10;
				tmpx11 = x11;
				tmpx12 = x12;
				tmpx20 = x20;
				tmpx21 = x21;
				tmpx22 = x22;

				if (inside)
				{
					vl = sqrt(1.0 - x02 * x02);
					geoang = atan(x02 / vl);

					guder = asinh(tan(geoang));

					newang = atan(sinh(guder - speed));

					proj0 = x00 / vl;
					proj1 = x01 / vl;

					rayon = pos2 / cos(geoang);
					pos2 = rayon * cos(newang);

					if (pos2 > beta)
					{
						newang = acos(beta / rayon);
						distrem = speed - guder + asinh(tan(newang));

						other = !other;

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = -mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = -nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = -nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////

						vl = sqrt(1.0 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						pos2 = rayon * cos(newang);

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////
					}
					else
					{
						if (pos2 < alpha)
						{
							inside = false;
							newang = -acos(alpha / rayon);
							distrem = speed - guder + asinh(tan(newang));
						}


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						if (!inside)
						{
							pos0 /= bigr;
							pos1 /= r;

							npos[0] = sin(pos0) * (bigr + r * cos(pos1));
							npos[1] = cos(pos0) * (bigr + r * cos(pos1));
							npos[2] = r * sin(pos1);

							tormat2(pos0, pos1, mat1);
							matflip2cpu(mat1, inv);
							matact2(inv, x00, x01, x02, nvecn);
							x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
							matact2(inv, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
							matact2(inv, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

							pos0 = npos[0];
							pos1 = npos[1];
							pos2 = npos[2];

							pos0 += x00 * distrem;
							pos1 += x01 * distrem;
							pos2 += x02 * distrem;
						}
					}
				}
				else
				{
					torcoll = toruscoll2(x00, pos0, x01, pos1, x02, pos2, r * r, bigr * bigr);
					if (torcoll == 65536 || torcoll > speed)
					{
						pos0 += x00 * speed;
						pos1 += x01 * speed;
						pos2 += x02 * speed;
					}
					else
					{
						tor0 = pos0 + torcoll * x00;
						tor1 = pos1 + torcoll * x01;
						tor2 = pos2 + torcoll * x02;
						xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

						theta = asin(tor2 / r);
						if (xyvec < bigr) theta = M_PI - theta;
						if (theta < 0) theta += 2.0 * M_PI;

						phi = acos(tor1 / xyvec);
						if (tor0 < 0) phi *= -1;
						if (phi < 0) phi += 2.0 * M_PI;

						tormat2(phi, theta, mat1);
						matinv2(mat1, inv);
						matflipcpu(inv, mat1);
						matact2(mat1, x00, x01, x02, nvecn);
						x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
						matact2(mat1, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat1, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 = phi * (bigr);
						pos1 = theta * r;
						pos2 = alpha;

						inside = true;

						distrem = speed - torcoll;

						vl = sqrt(1 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						rayon = pos2 / cos(geoang);
						pos2 = rayon * cos(newang);


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
					}
				}

				if (inside)
				{
					pos0 = fmod(pos0, 2.0 * M_PI * bigr);
					pos0 += signbit(pos0) * 2 * M_PI * bigr;

					pos1 = fmod(pos1, 2.0 * M_PI * r);
					pos1 += signbit(pos1) * 2.0 * M_PI * r;

					currx = floor(pos0 * nbx / (2.0 * M_PI * bigr));
					curry = floor(pos1 * nby / (2.0 * M_PI * r));
					currz = floor((pos2 - alpha) * nbz / (beta - alpha));

					if (other)
					{
						if (blocks2[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}
					else
					{
						if (blocks1[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}

				}


				newx00 = x00;
				newx01 = x01;
				newx02 = x02;

				x00 = x10;
				x01 = x11;
				x02 = x12;

				x10 = newx00;
				x11 = newx01;
				x12 = newx02;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			{
				newx00 = x00;
				newx01 = x01;
				newx02 = x02;

				x00 = x10;
				x01 = x11;
				x02 = x12;

				x10 = newx00;
				x11 = newx01;
				x12 = newx02;

				x00 = -x00;
				x01 = -x01;
				x02 = -x02;

				tmpinside = inside;
				tmpother = other;

				tmppos0 = pos0;
				tmppos1 = pos1;
				tmppos2 = pos2;

				tmpx00 = x00;
				tmpx01 = x01;
				tmpx02 = x02;
				tmpx10 = x10;
				tmpx11 = x11;
				tmpx12 = x12;
				tmpx20 = x20;
				tmpx21 = x21;
				tmpx22 = x22;

				if (inside)
				{
					vl = sqrt(1.0 - x02 * x02);
					geoang = atan(x02 / vl);

					guder = asinh(tan(geoang));

					newang = atan(sinh(guder - speed));

					proj0 = x00 / vl;
					proj1 = x01 / vl;

					rayon = pos2 / cos(geoang);
					pos2 = rayon * cos(newang);

					if (pos2 > beta)
					{
						newang = acos(beta / rayon);
						distrem = speed - guder + asinh(tan(newang));

						other = !other;

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = -mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = -nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = -nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////

						vl = sqrt(1.0 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						pos2 = rayon * cos(newang);

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////
					}
					else
					{
						if (pos2 < alpha)
						{
							inside = false;
							newang = -acos(alpha / rayon);
							distrem = speed - guder + asinh(tan(newang));
						}


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						if (!inside)
						{
							pos0 /= bigr;
							pos1 /= r;

							npos[0] = sin(pos0) * (bigr + r * cos(pos1));
							npos[1] = cos(pos0) * (bigr + r * cos(pos1));
							npos[2] = r * sin(pos1);

							tormat2(pos0, pos1, mat1);
							matflip2cpu(mat1, inv);
							matact2(inv, x00, x01, x02, nvecn);
							x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
							matact2(inv, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
							matact2(inv, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

							pos0 = npos[0];
							pos1 = npos[1];
							pos2 = npos[2];

							pos0 += x00 * distrem;
							pos1 += x01 * distrem;
							pos2 += x02 * distrem;
						}
					}
				}
				else
				{
					torcoll = toruscoll2(x00, pos0, x01, pos1, x02, pos2, r * r, bigr * bigr);
					if (torcoll == 65536 || torcoll > speed)
					{
						pos0 += x00 * speed;
						pos1 += x01 * speed;
						pos2 += x02 * speed;
					}
					else
					{
						tor0 = pos0 + torcoll * x00;
						tor1 = pos1 + torcoll * x01;
						tor2 = pos2 + torcoll * x02;
						xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

						theta = asin(tor2 / r);
						if (xyvec < bigr) theta = M_PI - theta;
						if (theta < 0) theta += 2.0 * M_PI;

						phi = acos(tor1 / xyvec);
						if (tor0 < 0) phi *= -1;
						if (phi < 0) phi += 2.0 * M_PI;

						tormat2(phi, theta, mat1);
						matinv2(mat1, inv);
						matflipcpu(inv, mat1);
						matact2(mat1, x00, x01, x02, nvecn);
						x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
						matact2(mat1, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat1, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 = phi * (bigr);
						pos1 = theta * r;
						pos2 = alpha;

						inside = true;

						distrem = speed - torcoll;

						vl = sqrt(1 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						rayon = pos2 / cos(geoang);
						pos2 = rayon * cos(newang);


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
					}
				}

				if (inside)
				{
					pos0 = fmod(pos0, 2.0 * M_PI * bigr);
					pos0 += signbit(pos0) * 2 * M_PI * bigr;

					pos1 = fmod(pos1, 2.0 * M_PI * r);
					pos1 += signbit(pos1) * 2.0 * M_PI * r;

					currx = floor(pos0 * nbx / (2.0 * M_PI * bigr));
					curry = floor(pos1 * nby / (2.0 * M_PI * r));
					currz = floor((pos2 - alpha) * nbz / (beta - alpha));

					if (other)
					{
						if (blocks2[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}
					else
					{
						if (blocks1[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}

				}



				x00 = -x00;
				x01 = -x01;
				x02 = -x02;

				newx00 = x00;
				newx01 = x01;
				newx02 = x02;

				x00 = x10;
				x01 = x11;
				x02 = x12;

				x10 = newx00;
				x11 = newx01;
				x12 = newx02;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			{
				x00 = -x00;
				x01 = -x01;
				x02 = -x02;

				tmpinside = inside;
				tmpother = other;

				tmppos0 = pos0;
				tmppos1 = pos1;
				tmppos2 = pos2;

				tmpx00 = x00;
				tmpx01 = x01;
				tmpx02 = x02;
				tmpx10 = x10;
				tmpx11 = x11;
				tmpx12 = x12;
				tmpx20 = x20;
				tmpx21 = x21;
				tmpx22 = x22;

				if (inside)
				{
					vl = sqrt(1.0 - x02 * x02);
					geoang = atan(x02 / vl);

					guder = asinh(tan(geoang));

					newang = atan(sinh(guder - speed));

					proj0 = x00 / vl;
					proj1 = x01 / vl;

					rayon = pos2 / cos(geoang);
					pos2 = rayon * cos(newang);

					if (pos2 > beta)
					{
						newang = acos(beta / rayon);
						distrem = speed - guder + asinh(tan(newang));

						other = !other;

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = -mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = -nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = -nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////

						vl = sqrt(1.0 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						pos2 = rayon * cos(newang);

						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						////////////////////////////////////////
					}
					else
					{
						if (pos2 < alpha)
						{
							inside = false;
							newang = -acos(alpha / rayon);
							distrem = speed - guder + asinh(tan(newang));
						}


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;

						if (!inside)
						{
							pos0 /= bigr;
							pos1 /= r;

							npos[0] = sin(pos0) * (bigr + r * cos(pos1));
							npos[1] = cos(pos0) * (bigr + r * cos(pos1));
							npos[2] = r * sin(pos1);

							tormat2(pos0, pos1, mat1);
							matflip2cpu(mat1, inv);
							matact2(inv, x00, x01, x02, nvecn);
							x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
							matact2(inv, x10, x11, x12, nvecn);
							x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
							matact2(inv, x20, x21, x22, nvecn);
							x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

							pos0 = npos[0];
							pos1 = npos[1];
							pos2 = npos[2];

							pos0 += x00 * distrem;
							pos1 += x01 * distrem;
							pos2 += x02 * distrem;
						}
					}
				}
				else
				{
					torcoll = toruscoll2(x00, pos0, x01, pos1, x02, pos2, r * r, bigr * bigr);
					if (torcoll == 65536 || torcoll > speed)
					{
						pos0 += x00 * speed;
						pos1 += x01 * speed;
						pos2 += x02 * speed;
					}
					else
					{
						tor0 = pos0 + torcoll * x00;
						tor1 = pos1 + torcoll * x01;
						tor2 = pos2 + torcoll * x02;
						xyvec = sqrt(tor0 * tor0 + tor1 * tor1);

						theta = asin(tor2 / r);
						if (xyvec < bigr) theta = M_PI - theta;
						if (theta < 0) theta += 2.0 * M_PI;

						phi = acos(tor1 / xyvec);
						if (tor0 < 0) phi *= -1;
						if (phi < 0) phi += 2.0 * M_PI;

						tormat2(phi, theta, mat1);
						matinv2(mat1, inv);
						matflipcpu(inv, mat1);
						matact2(mat1, x00, x01, x02, nvecn);
						x00 = nvecn[0]; x01 = nvecn[1]; x02 = nvecn[2];
						matact2(mat1, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat1, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 = phi * (bigr);
						pos1 = theta * r;
						pos2 = alpha;

						inside = true;

						distrem = speed - torcoll;

						vl = sqrt(1 - x02 * x02);
						geoang = atan(x02 / vl);

						guder = asinh(tan(geoang));

						newang = atan(sinh(guder - distrem));

						proj0 = x00 / vl;
						proj1 = x01 / vl;

						rayon = pos2 / cos(geoang);
						pos2 = rayon * cos(newang);


						mat[0] = x00;
						mat[3] = x01;
						mat[6] = x02;

						mat[1] = -sin(geoang) * proj0;
						mat[4] = -sin(geoang) * proj1;
						mat[7] = cos(geoang);

						mat[2] = mat[3] * mat[7] - mat[6] * mat[4];
						mat[5] = mat[6] * mat[1] - mat[0] * mat[7];
						mat[8] = mat[0] * mat[4] - mat[3] * mat[1];

						mat1[0] = cos(newang) * proj0;
						mat1[3] = cos(newang) * proj1;
						mat1[6] = sin(newang);

						mat1[1] = -sin(newang) * proj0;
						mat1[4] = -sin(newang) * proj1;
						mat1[7] = cos(newang);

						mat1[2] = mat[2];
						mat1[5] = mat[5];
						mat1[8] = mat[8];

						matinv2(mat, inv);
						matmult2(mat1, inv, mat);

						x00 = mat1[0];
						x01 = mat1[3];
						x02 = mat1[6];

						matact2(mat, x10, x11, x12, nvecn);
						x10 = nvecn[0]; x11 = nvecn[1]; x12 = nvecn[2];
						matact2(mat, x20, x21, x22, nvecn);
						x20 = nvecn[0]; x21 = nvecn[1]; x22 = nvecn[2];

						pos0 += rayon * (sin(geoang) - sin(newang)) * proj0;
						pos1 += rayon * (sin(geoang) - sin(newang)) * proj1;
					}
				}

				if (inside)
				{
					pos0 = fmod(pos0, 2.0 * M_PI * bigr);
					pos0 += signbit(pos0) * 2 * M_PI * bigr;

					pos1 = fmod(pos1, 2.0 * M_PI * r);
					pos1 += signbit(pos1) * 2.0 * M_PI * r;

					currx = floor(pos0 * nbx / (2.0 * M_PI * bigr));
					curry = floor(pos1 * nby / (2.0 * M_PI * r));
					currz = floor((pos2 - alpha) * nbz / (beta - alpha));

					if (other)
					{
						if (blocks2[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}
					else
					{
						if (blocks1[currx + nbx * curry + nbx * nby * currz])
						{
							inside = tmpinside;
							other = tmpother;

							pos0 = tmppos0;
							pos1 = tmppos1;
							pos2 = tmppos2;

							x00 = tmpx00;
							x01 = tmpx01;
							x02 = tmpx02;
							x10 = tmpx10;
							x11 = tmpx11;
							x12 = tmpx12;
							x20 = tmpx20;
							x21 = tmpx21;
							x22 = tmpx22;
						}
					}

				}


				x00 = -x00;
				x01 = -x01;
				x02 = -x02;
			}

        

        if (focus)
        {
			xl = sqrt(x00 * x00 + x01 * x01 + x02 * x02);
			x00 /= xl;
			x01 /= xl;
			x02 /= xl;

			dotp = x00 * x10 + x01 * x11 + x02 * x12;

			x10 = x10 - x00 * dotp;
			x11 = x11 - x01 * dotp;
			x12 = x12 - x02 * dotp;

			xl = sqrt(x10 * x10 + x11 * x11 + x12 * x12);
			x10 /= xl;
			x11 /= xl;
			x12 /= xl;

			dotp = x00 * x20 + x01 * x21 + x02 * x22;

			x20 = x20 - x00 * dotp;
			x21 = x21 - x01 * dotp;
			x22 = x22 - x02 * dotp;

			dotp = x10 * x20 + x11 * x21 + x12 * x22;

			x20 = x20 - x10 * dotp;
			x21 = x21 - x11 * dotp;
			x22 = x22 - x12 * dotp;

			xl = sqrt(x20 * x20 + x21 * x21 + x22 * x22);
			x20 /= xl;
			x21 /= xl;
			x22 /= xl;

            vec0 = dist * x00 + multy * x10 + multz * x20;
            vec1 = dist * x01 + multy * x11 + multz * x21;
            vec2 = dist * x02 + multy * x12 + multz * x22;

            addy0 = sqsz * x10;
            addy1 = sqsz * x11;
            addy2 = sqsz * x12;

            addz0 = -sqsz * x20;
            addz1 = -sqsz * x21;
            addz2 = -sqsz * x22;

			if (inside)
			{
				pos0 = fmod(pos0, 2.0*M_PI*bigr);
				pos0 += signbit(pos0)* 2 * M_PI * bigr;

				pos1 = fmod(pos1, 2.0 * M_PI * r);
				pos1 += signbit(pos1) * 2.0 * M_PI * r;

				currx = floor(pos0*nbx/(2.0 * M_PI * bigr));
				curry = floor(pos1 * nby / (2.0 * M_PI * r));
				currz= floor((pos2-alpha) * nbz  / (beta-alpha));
			}
       

            cudathingy(pixels, pos0, pos1, pos2, vec0, vec1, vec2, addy0, addy1, addy2, addz0, addz1, addz2, inside,alpha,beta,bigr,r,other, dx, dy, dz, currx, curry, currz, nbx, nby,nbz,blocks1,blocks2,rem,blockrand,reset);

			pixels[4 * (1920 * (1080/2) + (1920 / 2))] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920 / 2)) + 1] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920/2)) + 2] = 255;

			pixels[4 * (1920 * (1080 / 2+1) + (1920 / 2))] = 255;
			pixels[4 * (1920 * (1080 / 2+1) + (1920 / 2)) + 1] = 255;
			pixels[4 * (1920 * (1080 / 2+1) + (1920 / 2)) + 2] = 255;

			pixels[4 * (1920 * (1080 / 2-1) + (1920 / 2))] = 255;
			pixels[4 * (1920 * (1080 / 2-1) + (1920 / 2)) + 1] = 255;
			pixels[4 * (1920 * (1080 / 2-1) + (1920 / 2)) + 2] = 255;

			pixels[4 * (1920 * (1080 / 2) + (1920 / 2+1))] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920 / 2+1)) + 1] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920 / 2+1)) + 2] = 255;

			pixels[4 * (1920 * (1080 / 2) + (1920 / 2-1))] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920 / 2-1)) + 1] = 255;
			pixels[4 * (1920 * (1080 / 2) + (1920 / 2-1)) + 2] = 255;

            texture.update(pixels);
            sprite.setTexture(texture);
            window.draw(sprite);
            window.display();

			rem = false;
			blockrand = false;
			reset = false;
        }
    }

    cudaExit();
    return 0;
}

 double solvequartic2(double a0, double b0, double c0, double d0, double e0)
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

	return tmin;
}

 double toruscoll2(double a, double b, double c, double d, double e, double f, double m, double n)
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
	double abc = ab * c;

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

	return solvequartic2(t4, t3, t2, t1, t0);
}


 void tormat2(double phi, double theta, double* mat)
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

  double matdet2(double* m)
 {
	 return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
 }

 void matinv2(double* m, double* res)
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

 void matmult2(double* m1, double* m2, double* res)
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

 void matact2(double* m, double vecn0, double vecn1, double vecn2, double* nvecn)
 {
	 nvecn[0] = m[0] * vecn0 + m[1] * vecn1 + m[2] * vecn2;
	 nvecn[1] = m[3] * vecn0 + m[4] * vecn1 + m[5] * vecn2;
	 nvecn[2] = m[6] * vecn0 + m[7] * vecn1 + m[8] * vecn2;
 }

void matflipcpu(double* m, double* res)
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

  void matflip2cpu(double* m, double* res)
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


void setblocksrandcpu(bool* blocks1, bool* blocks2)
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

void setblockscpu(bool* blocks1, bool* blocks2)
{
	for (int i = 0; i < 30 * 30 * 30; i++)
	{
		blocks1[i] = true;
		blocks2[i] = true;
	}

}
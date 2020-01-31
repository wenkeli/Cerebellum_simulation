all: logviz qlearning

logviz:
	g++ -c -o logViz.o vizLogfile.cpp
	g++ -o logViz logViz.o -l SDL -l SDL_gfx -l SDL_ttf -l SDL_image 

qlearning:
	g++ -c -o cartpole.o cartpole.cpp
	g++ -o qLearning cartpole.o -l SDL -l SDL_image -l SDL_ttf -l SDL_gfx

clean:
	rm -f *.o logViz qLearning
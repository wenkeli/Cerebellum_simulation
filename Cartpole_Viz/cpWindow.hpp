#ifndef CP_WINDOW_HPP
#define CP_WINDOW_HPP

#include "SDL/SDL.h"
#include "SDL/SDL_gfxPrimitives.h"
#include "SDL/SDL_ttf.h"
#include "SDL/SDL_image.h"
#include <stdio.h>
#include <string>
#include <pthread.h>
#include <vector>

class MediaEventHandler {
public:
    virtual void handleMediaEvent(const int event) = 0;
};

class Button
{
public:
    Button(int x, int y, int w, int h,
           const SDL_Rect* _clip, const SDL_Rect* _mousedown_clip,
           const SDL_Rect* _thirdClip = NULL, const SDL_Rect* _fourthClip = NULL);
    void setSurfaces(SDL_Surface* screen, SDL_Surface* buttonSheet);
    //Handles events and set the button's sprite region
    // Returns true when the button is activated
    bool handle_events(const SDL_Event event);
    void show();

protected:
    SDL_Surface* screen;
    SDL_Surface* buttonSheet;

    // The area on screen that this button occupies
    SDL_Rect box;
    // The part of the button sprite sheet that will be shown
    SDL_Rect* clip1; SDL_Rect* clip2; SDL_Rect* clip3; SDL_Rect* clip4;

    bool pressed, inverted;
};

enum MEDIA_EVENTS {
    PREVIOUS = 0,
    BACK = 1,
    PLAYPAUSE = 2,
    FF = 3,
    NEXT = 4,
    SPEEDUP = 5,
    SLOWDOWN = 6
};

class CPWindow
{
public:
    CPWindow(float _tracklen, float _polelen, float _leftAngleBound, float _rightAngleBound,
             float lower_cartWidth);
    ~CPWindow();

    //void drawCartpole(CartPole* cp);
    void drawCartpole(float cartpos, float cartvel, float poleang, float polevel,
                      float lower_cartpos, float lower_cartvel, float lower_force, float lower_target,
                      float mz0Force, float mz1Force, bool errorLeft, bool errorRight,
                      int timeAloft, int trialNum, int cycle, float playspeed=1);

    void registerHandler(MediaEventHandler* handler) {
        handlers.push_back(handler);
    }

protected:
    void drawText(std::string text, SDL_Rect& dest);

    SDL_Surface* load_image(std::string filename);
    void SDL_Mainloop();
    static void *thread_helper(void *context) {
        ((CPWindow *)context)->SDL_Mainloop();
    }
    void renderScreen();

protected:
    float tracklen; // Length of track
    float polelen;  // Length of pole
    float leftAngBound;
    float rightAngBound;
    float lower_cartWidth;

    // Variables necessary for the drawing the cartpole image
    float cartpos, cartvel, poleang, polevel, mz0Force, mz1Force;
    float lower_cartpos, lower_cartvel, lower_force, lower_target;
    bool errorLeft, errorRight;
    uint bleedLeft, bleedRight; // These bleed the alpha channel of errors left/right
    int timeAloft, trialNum, cycle;
    float playspeed;

    SDL_Surface* screen;
    SDL_Surface* buttonSheet;

    SDL_Event event;
    TTF_Font *font;

    Button backButton, reverseButton, playButton, ffButton, nextButton, slowdownButton, speedupButton;
    pthread_t thread;

    // Event Handlers
    std::vector<MediaEventHandler*> handlers;

    //Screen attributes
    int screenWidth;
    int screenHeight;
    int screenBPP;

    // Locations of button in the buttons.jpg file
    static const SDL_Rect blue_back, blue_reverse, blue_play, blue_pause, blue_ff, blue_next,
        purple_reverse, purple_back, purple_ff, purple_next, purple_play, purple_pause,
        speedup, slowdown;
};

#endif

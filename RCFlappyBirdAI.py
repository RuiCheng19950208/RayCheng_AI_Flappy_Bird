# Reference: https://cloud.tencent.com/developer/article/1075527
import pygame
import os
import random
import pickle
import cv2
import pandas as pd
from pygame.locals import *
from sys import exit
from vector import Vector
import math
import operator
import numpy as np
import time
import copy
from hmmlearn import hmm
import sys

#--------------basic setting--------------------------

width=600
height=800
FLOOR = height+40
fps=120
AI_generation = 0
Pipe_born_x=1000
Pipe_born_interval=600
NN_init_range=2

# Game Modes
MODE_PLAYER = 0
MODE_AI_TRAINING = 1
MODE_PLAYER_VS_AI = 2
JUMP_VEL = -4.5
TITLE = "Ray Cheng AI Bird 2025"
#----------color---------
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(0,0,255)
YELLOW=(255,255,0)
GREEN=(0,255,0)
RED=(255,0,0)
GRAY=(128,128,128)
#-----------------------font---------------------------

# Create a function to get the correct resource path when using PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

font_name = resource_path("simusun.ttc")

#------------------------------------------------------

class Button:
    """
    Button class for UI interactions
    """
    def __init__(self, x, y, width, height, text, text_size=30, color=WHITE, hover_color=YELLOW, bg_color=(50, 50, 50)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.text_size = text_size
        self.color = color
        self.hover_color = hover_color
        self.bg_color = bg_color
        self.hover_bg_color = (70, 70, 70)
        self.is_hovered = False
        
    def draw(self, screen, font_name):
        # Draw button background with hover effect
        current_color = self.hover_color if self.is_hovered else self.color
        current_bg = self.hover_bg_color if self.is_hovered else self.bg_color
        
        # Draw button with background
        pygame.draw.rect(screen, current_bg, self.rect, 0, border_radius=12)
        pygame.draw.rect(screen, current_color, self.rect, 2, border_radius=12)
        
        # Draw button text
        font = pygame.font.Font(font_name, self.text_size)
        text_surface = font.render(self.text, True, current_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, mouse_pos):
        # Check if mouse is hovering over the button
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
        
    def is_clicked(self, mouse_pos, mouse_clicked):
        # Check if button is clicked
        return self.rect.collidepoint(mouse_pos) and mouse_clicked

def generate_new_model(model_list,n_iter,n_bird):
    variation_num=30
    bus_num=5
    def cross_breed(model_list):
        sex_machine=model_list[-1]
        choose_gene_from_bus_p = 0.3
        choose_p=np.array([1-choose_gene_from_bus_p,choose_gene_from_bus_p])

        new_weight1=np.zeros(sex_machine.weight1.shape)
        new_bias1 = np.zeros(sex_machine.bias1.shape)
        new_weight2 = np.zeros(sex_machine.weight2.shape)
        new_bias2= np.zeros(sex_machine.bias2.shape)

        bus = model_list[:-1]
        bus_num = len(bus)
        rand_index =np.random.randint(0, bus_num)
        chosen_bus=bus[rand_index]

        for i in range(sex_machine.weight1.shape[0]):
            for j in range(sex_machine.weight1.shape[1]):
                new_weight1[i][j]=np.random.choice([sex_machine.weight1[i][j],chosen_bus.weight1[i][j]], p=choose_p)
        for i in range(sex_machine.weight2.shape[0]):
            for j in range(sex_machine.weight2.shape[1]):
                new_weight2[i][j]=np.random.choice([sex_machine.weight2[i][j],chosen_bus.weight2[i][j]], p=choose_p)
        for i in range(sex_machine.bias1.shape[0]):
            for j in range(sex_machine.bias1.shape[1]):
                new_bias1[i][j]=np.random.choice([sex_machine.bias1[i][j],chosen_bus.bias1[i][j]], p=choose_p)
        for i in range(sex_machine.bias2.shape[0]):
            for j in range(sex_machine.bias2.shape[1]):
                new_bias2[i][j]=np.random.choice([sex_machine.bias2[i][j],chosen_bus.bias2[i][j]], p=choose_p)

        sex_machine.weight1 = new_weight1
        sex_machine.weight2 = new_weight2
        sex_machine.bias1 = new_bias1
        sex_machine.bias2 = new_bias2
        sex_machine.variation(False)

        return sex_machine

    copy_model_list=copy.deepcopy(model_list)
    for i in model_list:
        i.n_iter=n_iter
    new_model_list=[]
    for i in range(n_bird-variation_num-3):
        new_model_list.append(cross_breed(model_list[-bus_num:]))
        model_list[-bus_num:] = copy.deepcopy(copy_model_list[-bus_num:])
    for i in range(variation_num):
        new_model_list.append(model_list[-1].variation())  # add one model with noise
        model_list[-1] = copy.deepcopy(copy_model_list[-1])

    new_model_list.append(model_list[-1])  # Directly add best model
    new_model_list.append(model_list[-2])  # Directly add best model
    new_model_list.append(model_list[-3])  # Directly add best model



    return new_model_list


class RayCheng_NN(object):
    def __init__(self, n_input=4,n_hidden=10,n_iter=100):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_iter=n_iter
        self.weight1=(-NN_init_range/2) + NN_init_range*np.random.random(( self.n_input, self.n_hidden))
        self.bias1 = (-NN_init_range/2) + NN_init_range*np.random.random((1, self.n_hidden))
        self.weight2 =(-NN_init_range/2) +NN_init_range*np.random.random(( self.n_hidden,1))
        self.bias2=(-NN_init_range/2) +NN_init_range*np.random.random((1,1))
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def fit(self,input): # input np.array
        self.input = input
        step1 = self.sigmoid(np.dot(self.input,self.weight1)+self.bias1)
        output = self.sigmoid(np.dot(step1,self.weight2)+self.bias2)
        return output
    def variation(self,train=True):
        if train==True:
            up_limit=1+(2*self.n_iter/100)
            low_limit=1-(2*self.n_iter/100)
            self.weight1 =  np.multiply(self.weight1,np.random.uniform(low_limit,up_limit,(self.n_input, self.n_hidden)))
            self.bias1 = np.multiply(self.bias1,np.random.uniform(low_limit,up_limit,(1,self.n_hidden)))
            self.weight2 = np.multiply(self.weight2,np.random.uniform(low_limit,up_limit,(self.n_hidden,1)))
            self.bias2 = np.multiply(self.bias2,np.random.uniform(low_limit,up_limit,(1, 1)))
            self.weight1 +=  np.random.normal(loc=0.0, scale=0.001 * self.n_iter,size=(self.n_input, self.n_hidden))
            self.bias1 += np.random.normal(loc=0.0, scale=0.001 * self.n_iter, size=(1, self.n_hidden))
            self.weight2 += np.random.normal(loc=0.0, scale=0.001 * self.n_iter, size=(self.n_hidden,1))
            self.bias2 += np.random.normal(loc=0.0, scale=0.001 * self.n_iter, size=(1, 1))
        else:
            up_limit = 1 + (2 * 1 / 100)
            low_limit = 1 - (2 * 1 / 100)
            self.weight1 = np.multiply(self.weight1,np.random.uniform(low_limit, up_limit, (self.n_input, self.n_hidden)))
            self.bias1 = np.multiply(self.bias1, np.random.uniform(low_limit, up_limit, (1, self.n_hidden)))
            self.weight2 = np.multiply(self.weight2, np.random.uniform(low_limit, up_limit, (self.n_hidden, 1)))
            self.bias2 = np.multiply(self.bias2, np.random.uniform(low_limit, up_limit, (1, 1)))

        return self


class Player_bird(pygame.sprite.Sprite):
    """
    Bird class representing the flappy bird
    """
    def __init__(self,game, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """

        self.jump_vel = JUMP_VEL
        self.groups = game.all_sprites,game.player_bird_group
        pygame.sprite.Sprite.__init__(self,self.groups)
        self.game=game

        self.ANIMATION_TIME = 5
        self.x = x
        self.y = y
        self.gravity_a = self.game.gravity_a
        self.vel = 0
        self.img_count = 0
        self.IMGS = self.game.bird_images
        self.image = self.IMGS[0]

        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centery = y
        self.rect.centerx = x


    def get_key(self):
        keys=pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            self.jump()
    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = self.jump_vel

    def move(self):
        """
        make the bird move
        :return: None
        """
        # for downward acceleration
        self.vel += self.gravity_a * self.game.time_scale **2
        self.y += self.vel

    def update(self):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        self.get_key()
        self.img_count += 1
        if self.y<=0:
            self.y=0
            self.vel=0

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.image = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.image = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.image = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.image = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.image = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping

        # tilt the bird
        self.rect.y = self.y

        self.move()

class AI_bird(pygame.sprite.Sprite):
    """
    Bird class representing the flappy bird
    """
    def __init__(self,game, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """

        self.jump_vel = JUMP_VEL
        self.groups = game.all_sprites,game.AI_bird_group
        pygame.sprite.Sprite.__init__(self,self.groups)
        self.game=game
        self.AI_NN = RayCheng_NN(n_iter=self.game.iteration_time)
        self.ANIMATION_TIME = 5
        self.x = x
        self.y = y
        self.gravity_a = self.game.gravity_a
        self.vel = 0.0
        self.img_count = 0
        self.IMGS = self.game.bird_images_AI
        for i in self.IMGS:
            i.set_alpha(70)
        self.image = self.IMGS[0]
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centery = y
        self.rect.centerx = x


    def get_key(self):
        # thresh= random.uniform(5, 30)* self.game.time_scale
        for i in range(len(self.game.top_pipes_group.sprites())):
            if self.game.top_pipes_group.sprites()[i].passed==False:
                input= np.array(
                    [self.vel/(20 * self.game.time_scale),
                     (self.game.top_pipes_group.sprites()[i].rect.centerx - self.rect.centerx) / Pipe_born_interval,
                     ( self.game.top_pipes_group.sprites()[i].rect.centery -self.rect.centery) / height,
                     ( self.rect.centery-self.game.bot_pipes_group.sprites()[i].rect.centery) / height])
                # print(input)
                output=self.AI_NN.fit(input)
                # print(output)
                if output >= 0.5:
                    self.jump()

                break

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = self.jump_vel * self.game.time_scale

    def move(self):
        """
        make the bird move
        :return: None
        """
        # for downward acceleration
        self.vel += self.gravity_a * (self.game.time_scale**2)
        self.y += self.vel

    def update(self):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """

        self.get_key()
        self.img_count += 1
        if self.y<=0:
            self.y=0
            self.vel=0
        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.image = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.image = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.image = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.image = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.image = self.IMGS[0]
            self.img_count = 0
        # so when bird is nose diving it isn't flapping
        # tilt the bird
        self.rect.y = self.y
        self.move()

class Pipe_bot(pygame.sprite.Sprite):
    """
    represents a pipe object
    """
    def __init__(self, game,x,y):
        self.groups = game.all_sprites, game.pipes_group,game.bot_pipes_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.vel = self.game.x_vel
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(self.game.pipe_img, False, True)
        self.image = self.game.pipe_img
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y


    def update(self):
        self.rect.x -= self.vel * self.game.time_scale

class Pipe_top(pygame.sprite.Sprite):
    """
    represents a pipe object
    """
    def __init__(self, game,x,y):
        self.groups = game.all_sprites, game.pipes_group,game.top_pipes_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.GAP = 300
        self.vel = self.game.x_vel
        self.image = pygame.transform.flip(self.game.pipe_img, False, True)
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y - self.GAP - self.image.get_height()
        self.passed = False

    def update(self):
        self.rect.x -= self.vel * self.game.time_scale
        if self.game.mode in [0,2]:
            if self.rect.x+60<= self.game.player.x and self.passed ==False:
                self.passed =True
                self.game.random_next_pipe_heigh = random.randrange(height-150, height+150)
                self.add_pipe()
        elif self.game.mode==1:
            if self.rect.x+60 <= self.game.AI_bird_group.sprites()[-1].x and self.passed == False:
                self.passed = True
                self.game.random_next_pipe_heigh = random.randrange(height - 150, height + 150)
                self.add_pipe()


    def add_pipe(self):
        self.game.score +=1
        pipe_bot=Pipe_bot(self.game,self.rect.x+2*Pipe_born_interval,self.game.random_next_pipe_heigh)
        pipe_top=Pipe_top(self.game,self.rect.x+2*Pipe_born_interval,self.game.random_next_pipe_heigh)
        self.game.pipes.append(pipe_bot)
        self.game.pipes.append(pipe_top)
    def kill_self(self):
        if self.rect.x<-200:
            self.kill()


class Base(pygame.sprite.Sprite):
    """
    Represnts the moving floor of the game
    """
    def __init__(self, game, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.groups = game.all_sprites, game.base_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.image = self.game.base_img
        self.time_scale = self.game.time_scale
        self.vel = self.game.x_vel
        self.WIDTH = self.game.base_img.get_width()

        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centery = y
        self.rect.centerx = (self.x1+self.x2)/2

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.vel * self.game.time_scale
        self.x2 -= self.vel * self.game.time_scale
        if self.x1 + self.x2 < 0:
            self.x1 = 0
            self.x2=self.WIDTH

    def update(self):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        self.move()
        self.rect.centery = self.y
        self.rect.centerx = (self.x1 + self.x2) / 2


class Game():
    def __init__(self):
        # game window initialize
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.font_name = pygame.font.match_font(font_name)
        self.snd_dir = 'sounds'
        self.img_dir = 'imgs'
        self.music_init(self.snd_dir)
        self.img_init(self.img_dir)
        self.AI_birds_num = 50
        self.Born_location = [width/5, height/2]
        self.time_scale_list = [0.5, 1, 2, 4]
        self.time_scale_index = 1  
        self.time_scale = self.time_scale_list[self.time_scale_index]
        self.x_vel = 3.0  # Reduced from 4.0
        self.gravity_a = 0.15  # Reduced from 0.2
        self.score = 0
        self.random_next_pipe_heigh = random.randrange(height-150, height+150)
        self.mode = MODE_PLAYER  # Default to player mode
        self.iteration_time_init = 200
        self.iteration_time = self.iteration_time_init
        self.model_list = []
        self.AI_player = 3
        self.player_defeated = False
        self.ai_defeated = False
        
        # Create mode selection buttons
        button_width = 300
        button_height = 50
        button_x = width//2 - button_width//2
        self.player_button = Button(button_x, (height/4) + 140, button_width, button_height, "Player Mode", 30)
        self.ai_training_button = Button(button_x, (height/4) + 200, button_width, button_height, "AI Training Mode", 30)
        self.vs_ai_button = Button(button_x, (height/4) + 260, button_width, button_height, "Player vs AI Mode", 30)

    def music_init(self,snd_dir):
        self.BGM=pygame.mixer.music.load(resource_path(os.path.join(snd_dir,'game_music.wav')))
        self.GameOverSound = pygame.mixer.Sound(resource_path(os.path.join(snd_dir, 'game_over.wav')))


    def img_init(self,img_dir):
        self.pipe_img = pygame.transform.scale2x(pygame.image.load(resource_path(os.path.join(img_dir, "pipe.png"))).convert_alpha())
        self.bg_img = pygame.transform.scale(pygame.image.load(resource_path(os.path.join(img_dir, "bg.png"))).convert_alpha(), (600, 900))
        self.bird_images = [pygame.transform.scale2x(pygame.image.load(resource_path(os.path.join(img_dir, "bird" + str(x) + ".png")))) for x in range(1, 4)]
        self.bird_images_AI = [pygame.transform.scale2x(pygame.image.load(resource_path(os.path.join(img_dir, "bird" + str(x) + ".png")))) for x in range(1, 4)]
        self.base_img = pygame.transform.scale(pygame.image.load(resource_path(os.path.join(img_dir, "base.png"))).convert_alpha(),(2*width,300))

    # def loaddata(self):
    #     game_folder = os.path.dirname(__file__)
    #     self.map_data = []
    #     with open(os.path.join(game_folder,'MapEditor.txt'),'rt') as f:
    #         for line in f:
    #             self.map_data.append(line)
    def new(self):
        #restart
        self.all_sprites = pygame.sprite.Group()
        self.player_bird_group = pygame.sprite.Group()
        self.AI_bird_group = pygame.sprite.Group()
        self.pipes_group = pygame.sprite.Group()
        self.top_pipes_group = pygame.sprite.Group()
        self.bot_pipes_group = pygame.sprite.Group()
        self.base_group = pygame.sprite.Group()
        self.base = Base(self,FLOOR)
        self.player_defeated = False
        self.ai_defeated = False

        self.pipes=[]
        self.pipes.append(Pipe_bot(self,Pipe_born_x,self.random_next_pipe_heigh))
        self.pipes.append(Pipe_top(self, Pipe_born_x,self.random_next_pipe_heigh))
        self.pipes.append(Pipe_bot(self, Pipe_born_interval + Pipe_born_x, self.random_next_pipe_heigh))
        self.pipes.append(Pipe_top(self, Pipe_born_interval + Pipe_born_x, self.random_next_pipe_heigh))

        # Set appropriate time scale for each mode
        if self.mode == MODE_PLAYER:
            self.time_scale_index = 1  # Set to 1x speed
            self.time_scale = self.time_scale_list[self.time_scale_index]
            self.player = Player_bird(self, self.Born_location[0], self.Born_location[0])
        elif self.mode == MODE_AI_TRAINING:
            # Set to fastest speed for training
            self.time_scale_index = len(self.time_scale_list) - 1  # Set to maximum speed (8x)
            self.time_scale = self.time_scale_list[self.time_scale_index]
            
            # Create birds first
            for i in range(self.AI_birds_num):
                AI_bird(self, self.Born_location[0], self.Born_location[0])
            
            # Then assign models safely
            if len(self.model_list) > 0:
                model_count = len(self.model_list)
                for i, bird in enumerate(self.AI_bird_group.sprites()):
                    if i < model_count:
                        bird.AI_NN = self.model_list[i]
                    else:
                        # For birds without models, either use the best model or leave as default
                        if model_count > 0:
                            # Copy the best model (assumed to be the last one)
                            bird.AI_NN = copy.deepcopy(self.model_list[-1])
                            # Add some variation
                            bird.AI_NN.variation(True)
            
            self.model_list=[]

        elif self.mode == MODE_PLAYER_VS_AI:
            self.time_scale_index = 1  # Set to 1x speed
            self.time_scale = self.time_scale_list[self.time_scale_index]
            self.player = Player_bird(self, self.Born_location[0], self.Born_location[0])

            AI_bird(self, self.Born_location[0], self.Born_location[0])
            model_path = resource_path('AI_bird_model.txt')
            if os.path.exists(model_path):
                with open(model_path, "rb") as fp:  # Unpickling
                    self.AI_bird_group.sprites()[-1].AI_NN = pickle.load(fp)

        self.run()

    def run(self):
        # game loop
        # pygame.mixer.music.play(loops=-1)
        self.playing = True
        self.score = 0
        while self.playing:
            # self.dt = self.clock.tick(fps)/1000
            self.events()
            self.update()
            self.draw()

    def update(self):
       #loop update
        self.all_sprites.update()

        # for b in self.bullets:
        #     if pygame.sprite.spritecollide(b, self.bricks, True):
        #         b.kill()
        #     if pygame.sprite.spritecollide(b, self.walls, False):
        #         b.kill()
        #     if pygame.sprite.spritecollide(b, self.enemy, True):
        #         b.kill()
        #         self.enemykill_sound.play()

        if self.mode in [MODE_PLAYER, MODE_PLAYER_VS_AI]:
            if pygame.sprite.spritecollide(self.player, self.base_group, False) or pygame.sprite.spritecollide(self.player, self.pipes_group, False):
                if self.mode == MODE_PLAYER_VS_AI:
                    self.player_defeated = True
                self.player.kill()
                pygame.mixer.music.stop()
                self.GameOverSound.play()
                self.playing = False

        if self.mode in [MODE_AI_TRAINING, MODE_PLAYER_VS_AI]:
            for i in self.AI_bird_group.sprites():
                if pygame.sprite.spritecollide(i, self.base_group, False) or pygame.sprite.spritecollide(i, self.pipes_group, False):
                    if self.mode == MODE_PLAYER_VS_AI:
                        self.ai_defeated = True
                    self.model_list.append(i.AI_NN)
                    i.kill()

        if self.mode == MODE_AI_TRAINING:
            if self.iteration_time>0 and len(self.AI_bird_group.sprites())==0:
                self.iteration_time -=1
                self.model_list=generate_new_model(self.model_list,max(self.iteration_time-100,1),self.AI_birds_num)
                print('All bot dead')
                self.new()
            elif self.iteration_time==0  and len(self.AI_bird_group.sprites())==0:
                self.model_list = generate_new_model(self.model_list, max(self.iteration_time-100,1), self.AI_birds_num)
                with open(resource_path('AI_bird_model.txt'), "wb") as fp:  # Pickling
                    pickle.dump(self.model_list[-1], fp)
                print('AI_bird model saved! ')
                pygame.mixer.music.stop()
                self.iteration_time = self.iteration_time_init
                self.GameOverSound.play()
                self.playing = False
            elif self.score>=70:
                with open(resource_path('AI_bird_model.txt'), "wb") as fp:  # Pickling
                    pickle.dump(self.AI_bird_group.sprites()[-1].AI_NN, fp)
                print('AI_bird model saved! ')
                self.GameOverSound.play()
                self.playing = False





    def events(self):
        #define trigger
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
                if self.playing:
                    self.playing = False
                self.running = False
            if pygame.key.get_pressed()[pygame.K_RIGHT]:
                if self.playing:
                    # Allow speed control in all modes
                    self.time_scale_index = min(self.time_scale_index+1, len(self.time_scale_list)-1)
                    self.time_scale = self.time_scale_list[self.time_scale_index]
                    print('Time scale: x' + str(self.time_scale))

            if pygame.key.get_pressed()[pygame.K_LEFT]:
                if self.playing:
                    # Allow speed control in all modes
                    self.time_scale_index = max(self.time_scale_index-1, 0)
                    self.time_scale = self.time_scale_list[self.time_scale_index]
                    print('Time scale: x' + str(self.time_scale))



    def draw(self):

        # draw in the window
        self.screen.fill(BLACK)
        self.screen.blit(self.bg_img, (0,0))
        self.all_sprites.draw(self.screen)
        self.draw_text_left_top('Author:Ray Cheng', 30, WHITE, 5, 0)
        self.draw_text_left_top('Score: '+str(self.score), 30, WHITE, 5, 32)
        self.draw_text_left_top('Time scale: x' + str(self.time_scale), 30, WHITE, 5, 64)
        
        if self.mode == MODE_AI_TRAINING:
            self.draw_text_left_top('Iteration last: ' + str(self.iteration_time), 30, WHITE, 5, 96)
            
        if self.mode == MODE_PLAYER_VS_AI:
            if len(self.AI_bird_group.sprites()) == 0:
                self.draw_text_left_top("YOU WIN!", 30, GREEN, 5, 96)
                AI_score = self.score
                self.draw_text_left_top("AI score: "+str(AI_score), 30, WHITE, 5, 128)
            elif self.player_defeated and not self.playing:
                self.draw_text_left_top("YOU LOSE TO AI!", 30, RED, 5, 96)
                self.draw_text_left_top("Your score: "+str(self.score), 30, WHITE, 5, 128)

        pygame.display.flip()

    def draw_text(self,text,size,color,x,y):
        font=pygame.font.Font(self.font_name,size)
        text_surface = font.render(text,True,color)
        text_rect= text_surface.get_rect()
        text_rect.midtop=(x,y)
        self.screen.blit(text_surface,text_rect)

    def draw_text_left_top(self,text,size,color,x,y):
        font=pygame.font.Font(self.font_name,size)
        text_surface = font.render(text,True,color)
        text_rect= text_surface.get_rect()
        text_rect.topleft=(x,y)
        self.screen.blit(text_surface,text_rect)

    def show_start_screen(self):
        self.screen.fill(BLACK)
        self.draw_text(TITLE, 50, WHITE, width/2, height/4)
        
        # Draw mode selection buttons
        self.player_button.draw(self.screen, self.font_name)
        self.ai_training_button.draw(self.screen, self.font_name)
        self.vs_ai_button.draw(self.screen, self.font_name)
        
        self.draw_text('Press Space or Up to jump!', 30, YELLOW, width/2, (height/4) + 320)
        pygame.display.flip()
        
        # Wait for player to select a mode
        waiting = True
        while waiting and self.running:
            self.clock.tick(fps)
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = False
            
            # Check button hover states
            self.player_button.check_hover(mouse_pos)
            self.ai_training_button.check_hover(mouse_pos)
            self.vs_ai_button.check_hover(mouse_pos)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
                    self.running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_clicked = True
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_PLAYER
                    elif event.key == pygame.K_w:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_AI_TRAINING
                    elif event.key == pygame.K_e:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_PLAYER_VS_AI
            
            # Check for button clicks
            if mouse_clicked:
                if self.player_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_PLAYER
                elif self.ai_training_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_AI_TRAINING
                elif self.vs_ai_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_PLAYER_VS_AI
            
            # Update the display to show hover effects
            if any([self.player_button.is_hovered, self.ai_training_button.is_hovered, self.vs_ai_button.is_hovered]):
                self.screen.fill(BLACK)
                self.draw_text(TITLE, 50, WHITE, width/2, height/4)
                self.player_button.draw(self.screen, self.font_name)
                self.ai_training_button.draw(self.screen, self.font_name)
                self.vs_ai_button.draw(self.screen, self.font_name)
                self.draw_text('Press Space or Up to jump!', 30, YELLOW, width/2, (height/4) + 320)
                pygame.display.flip()

    def show_gameover_screen(self):
        if not self.running:
            return

        waiting = True
        
        # Helper function to draw game over screen content
        def draw_gameover_content():
            self.screen.fill(BLACK)
            
            if self.mode == MODE_PLAYER_VS_AI:
                if self.ai_defeated:
                    self.draw_text('YOU WIN!', 50, GREEN, width/2, height/4)
                elif self.player_defeated:
                    self.draw_text('YOU LOSE TO AI!', 50, RED, width/2, height/4)
                else:
                    self.draw_text('Game Over', 50, WHITE, width/2, height/4)
            else:
                self.draw_text('Game Over', 50, WHITE, width/2, height/4)
            
            # Draw final score
            self.draw_text(f'Your Score: {self.score}', 30, WHITE, width/2, height/4 + 60)
            
            # Draw mode selection buttons
            self.player_button.draw(self.screen, self.font_name)
            self.ai_training_button.draw(self.screen, self.font_name)
            self.vs_ai_button.draw(self.screen, self.font_name)
            
            pygame.display.flip()
        
        # Initial draw of game over screen
        draw_gameover_content()
        
        # Wait for button click or key press in a separate loop
        while waiting and self.running:
            self.clock.tick(fps)
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = False
            
            # Check button hover states
            hover_changed = False
            for button in [self.player_button, self.ai_training_button, self.vs_ai_button]:
                was_hovered = button.is_hovered
                button.check_hover(mouse_pos)
                if was_hovered != button.is_hovered:
                    hover_changed = True
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
                    self.running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_clicked = True
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_PLAYER
                    elif event.key == pygame.K_w:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_AI_TRAINING
                    elif event.key == pygame.K_e:
                        waiting = False
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(0.3)
                        self.mode = MODE_PLAYER_VS_AI
            
            # Check for button clicks
            if mouse_clicked:
                if self.player_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_PLAYER
                elif self.ai_training_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_AI_TRAINING
                elif self.vs_ai_button.is_clicked(mouse_pos, mouse_clicked):
                    waiting = False
                    pygame.mixer.music.play(-1)
                    pygame.mixer.music.set_volume(0.3)
                    self.mode = MODE_PLAYER_VS_AI
            
            # Update the display to show hover effects (only if hover state changed)
            if hover_changed:
                draw_gameover_content()

    def get_image(self,img, x, y, w, h):
        image = pygame.Surface((w, h))
        image.blit(img, (0, 0), (x, y, w, h))
        # image=pygame.transform.scale(image,(w//2,h//2)) #If you want to resize
        return image
g=Game()
g.show_start_screen()
while g.running:
    g.new()
    if g.running:  # Only show game over screen if not quitting
        g.show_gameover_screen()

pygame.quit()


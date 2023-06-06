import pygame

class Speak(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        for num in range (0,15):
            img = pygame.image.load(f'library/images/agent/speaking/speak.{num}.png')
            img = pygame.transform.scale(img, (720,720))
            self.images.append(img)
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [360,360]
        self.counter = 0

    def update(self):
        speak_speed = 1
        # update speak animation
        self.counter += 1

        if self.counter >= speak_speed and self.index < len(self.images)-1:
            self.counter = 0
            self.index += 1
            self.image = self.images[self.index]

        # if the animation is complete, reset animation index
        if self.index >= len(self.images) - 1 and self.counter >= speak_speed:
            self.kill()

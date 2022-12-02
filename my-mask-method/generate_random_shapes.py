import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
from torchvision import transforms
import torch


bernstein = lambda n, k, t: binom(n, k) * t**k * (1.0 - t)**(n - k)

def bezier(points, num=100):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, r=0.25, num_points=100):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        d = np.sqrt(np.sum((self.p2 - self.p1)**2))
        self.r = r * d
        self.num_points = num_points
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points()

    def calc_intermediate_points(self):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1), self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2 + np.pi), self.r*np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.num_points)


def get_curve(points, r=0.25, num_points=100):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2], points[i+1,2], r=r, num_points=num_points)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p, axis=0)
    s = np.arctan2(d[:,1], d[:,0])
    return p[np.argsort(s),:]

def get_bezier_curve(a, p=0.52, r=0.25, num_points=100):
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    norm = np.linalg.norm(d, axis=1)[:, np.newaxis] 
    d = d / norm
    d2 = np.roll(d, 1, axis=0)
    d = (p * d) + ((1 - p) * d2)
    ang = np.arctan2(d[:,1], d[:,0])
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=r, num_points=num_points)
    return c, a


def get_random_points(n=6, height=1.0, width=1.0, mindst=None, rec=200):
    if mindst is None:
        mindst = min(height, width) / n
    
    while rec > 0:
        x = np.random.rand(n, 1) * width
        y = np.random.rand(n, 1) * height

        a = np.concatenate([x, y], axis=1)
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)

        d = np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1))

        a = a[:-1]

        if np.all(d >= mindst):
            return a
        else:
            rec -= 1

    return a


def get_random_shape(hw, height_range=(0.2, 0.5), width_range=(0.7, 0.9), n=6, p=0.52, r=0.25, num_points=20):
    img_h, img_w = hw

    h = random.randint(int(img_h * height_range[0]), int(img_h * height_range[1]))
    w = random.randint(int(img_w * width_range[0]), int(img_w * width_range[1]))

    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)

    points = get_random_points(n=n, height=h, width=w)
    points += np.array([x, y])

    curve, points = get_bezier_curve(points, p=p, r=r, num_points=num_points)

    curve = list(map(tuple, curve))
    color = tuple((np.random.rand(3) * 255).astype(int))

    img = Image.new(mode='RGB', size=(img_w, img_h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon(curve, fill=color)

    tensor_img = transforms.functional.to_tensor(img)

    mask = torch.mean(tensor_img, dim=0, keepdim=True)
    mask[mask != 0] = 1
    mask = 1 - mask

    return tensor_img, mask


if __name__=='__main__':

    import time

    start = time.time()

    for i in range(128):
        get_random_shape((384, 128))

    end = time.time()

    print(end-start)

    exit()

    '''fig, ax = plt.subplots()
    ax.set_aspect("equal")

    lin = 0.1

    for c in np.array([[0,0], [0,1], [1,0], [1,1]]):

        a = get_random_points() + c
        curve, a = get_bezier_curve(a)
        x, y = curve.T
        plt.plot(x, y)

        for point in a:
            p = point[:2] + np.array([lin*np.cos(point[2]), lin*np.sin(point[2])])
            plt.plot([point[0], p[0]], [point[1], p[1]], color="limegreen")


    plt.show()'''


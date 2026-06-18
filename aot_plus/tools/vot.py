"""
\file vot.py

@brief Python utility functions for VOT toolkit integration

@author Luka Cehovin Zajc, Alessio Dore

@date 2026

"""

import os
import collections
import numpy as np

_USE_TRAX = os.environ.get("VOT_USE_TRAX", "1") == "1"

try:
    import trax
    
    if trax._ctypes.trax_version().decode("ascii") < "4.0.0" and _USE_TRAX:
        raise ImportError('TraX version 4.0.0 or newer is required.')
except ImportError:
    _USE_TRAX = False

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])
Empty = collections.namedtuple('Empty', [])

def _parse_region(line):
    """ Parse a region from a line of text, the format can be either rectangle, polygon or mask. """
    line = line.strip()
    if line[0] == 'm':
        # input is a mask - decode it
        encoded = [int(x) for x in line[1:].split(",")]
        ox, oy, width, height = encoded[:4]

        v = np.zeros(width * height, dtype=np.uint8)

        # set id of the last different element to the beginning of the vector
        idx_ = 0
        for i in range(len(encoded[4:])):
            if i % 2 != 0:
                # write as many 1s as RLE says (zeros are already in the vector)
                for j in range(encoded[4+i]):
                    v[idx_+j] = 1
            idx_ += encoded[4+i]
        
        v = v.reshape((height, width))
        # Pad the mask to the original size of the image, the offset is given by (ox, oy)
        m_ = np.zeros((oy + height, ox + width), dtype=np.uint8)
        m_[oy:oy+height, ox:ox+width] = v
        
        return m_
    else:
        # input is not a mask - check if special, rectangle or polygon
        tokens = [float(t) for t in line.split(",")]
        if len(tokens) == 1:
            return Empty()
        if len(tokens) == 2:
            return Point(tokens[0], tokens[1])
        if len(tokens) == 4:
            return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
        elif len(tokens) % 2 == 0 and len(tokens) > 4:
            return Polygon([(x_, y_) for x_, y_ in zip(tokens[::2], tokens[1::2])])
    return None

def _encode_region(region):
    
    if isinstance(region, Empty):
        return "0"
    elif isinstance(region, Point):
        return f"{region.x},{region.y}"
    elif isinstance(region, Rectangle):
        return f"{region.x},{region.y},{region.width},{region.height}"
    elif isinstance(region, Polygon):
        return ",".join(f"{point.x},{point.y}" for point in region.points)
    elif isinstance(region, np.ndarray):
        ys, xs = np.nonzero(region)
        if len(xs) == 0 or len(ys) == 0: return "0"
        ox, oy = xs.min(), ys.min()
        width, height = xs.max() - ox + 1, ys.max() - oy + 1
        v = region[oy:oy+height, ox:ox+width].flatten()
        rle = []
        count = 1
        for i in range(1, len(v)):
            if v[i] != v[i-1]:
                rle.append(count)
                count = 1
            else:
                count += 1
        rle.append(count)

        return f"m{ox},{oy},{width},{height}," + ",".join(str(x) for x in rle)

def _validate_region(region, valid_formats):
    if isinstance(region, Empty):
        return "empty" in valid_formats
    elif isinstance(region, Point):
        return "point" in valid_formats
    elif isinstance(region, Rectangle):
        return "rectangle" in valid_formats
    elif isinstance(region, Polygon):
        return "polygon" in valid_formats
    elif isinstance(region, np.ndarray):
        return "mask" in valid_formats
    else:
        return False

class VOT(object):
    """ Base class for VOT toolkit integration in Python.
        This class is only a wrapper around the TraX protocol and can be used for single or multi-object tracking.
        The wrapper assumes that the experiment will provide new objects onlf at the first frame and will fail otherwise."""
    def __init__(self, region_format, channels=None, multiobject: bool = None):
        """ Constructor for the VOT wrapper.

        Args:
            region_format: Region format options
            channels: Channels that are supported by the tracker
            multiobject: Whether to use multi-object tracking
        """
        if _USE_TRAX:
            assert(region_format in ["rectangle", "polygon", "mask"])
        else:
            assert(region_format in ["rectangle", "polygon", "mask", "point"])

        if multiobject is None:
            multiobject = os.environ.get('VOT_MULTI_OBJECT', '0') == '1'

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise RuntimeError(f'Illegal configuration {channels}.')

        self._trax = None
        self._multiobject = multiobject    
        
        if _USE_TRAX:
    
            self._trax = trax.Server([region_format], ["path"], channels, metadata=dict(vot="python"), multiobject=multiobject)

            request = self._trax.wait()
            assert(request.type == 'initialize')

            self._objects = []

            assert len(request.objects) > 0 and (multiobject or len(request.objects) == 1)

            for tobject, _ in request.objects:
                if isinstance(tobject, trax.Polygon):
                    self._objects.append(Polygon([Point(x[0], x[1]) for x in tobject]))
                elif isinstance(tobject, trax.Mask):
                    self._objects.append(tobject.array(True))
                else:
                    self._objects.append(Rectangle(*tobject.bounds()))

            self._image = [x.path() for k, x in request.image.items()]
            if len(self._image) == 1:
                self._image = self._image[0]

            self._trax.status(request.objects)
            
        else:
            
            # Fallback to folder protocol if TraX is not available, this is useful for testing and debugging without TraX
            self._frames = []
            self._objects = []
            self._object_keys = []
            self._object_trajectory = []
            
            frames = []
            for _, channel in enumerate(channels):
                filename = f"frames_{channel}.txt"
                if not os.path.exists(filename):
                    raise RuntimeError(f"Missing frames file for channel {channel}")
               
                with open(filename, "r", encoding="utf-8") as f:
                    frames.append([line.strip() for line in f])
               
            if len(frames) == 1:
                self._frames = frames[0]
            else:
                assert all(len(f) == len(frames[0]) for f in frames), "All frames files must have the same number of lines"
                self._frames = list(zip(*frames))
                    
            # List all files following the pattern query_*.txt in the current folder
            queries = [f for f in os.listdir() if f.startswith("query_") and f.endswith(".txt")]
        
            assert len(queries) > 0, "No query file found"
        
            # Read image list from specified file    

            for query_file in queries:
                with open(query_file, "r", encoding="utf-8") as f:
                    object_id = query_file[len("query_"):-len(".txt")]
                    lines = [line.strip() for line in f if line.strip()]
                    offset = int(lines[0])
                    if offset != 0:
                        raise RuntimeError(f"Only offset 0 is supported in the wrapper, but got offset {offset} in file {query_file}")
                    state = _parse_region(lines[1])
                    if not _validate_region(state, [region_format]):
                        raise RuntimeError(f"Invalid region format in file {query_file}")
                    properties = {}
                    for line in lines[2:]:
                        if "=" in line:
                            key, value = line.split("=", 1)
                            properties[key] = value
                    self._objects.append(state)
                    self._object_keys.append(object_id)
                    self._object_trajectory.append([state])
                    
            self._position = 0
            
    def region(self):
        """
        Returns initialization region for the first frame in single object tracking mode.

        Returns:
            initialization region
        """

        assert not self._multiobject

        return self.objects()[0]

    def objects(self):
        """
        Returns initialization regions for the first frame in multi object tracking mode.

        Returns:
            initialization regions for all objects
        """

        return self._objects

    def report(self, status):
        """
        Report the tracking results to the client

        Arguments:
            status: region for the frame or a list of regions in case of multi object tracking
        """

        def convert_trax(region):
            """ Convert region to TraX format """
            # If region is None, return empty region
            if region is None: return trax.Rectangle.create(0, 0, 0, 0)
            assert isinstance(region, (Empty, Rectangle, Polygon, np.ndarray))
            if isinstance(region, Empty):
                return trax.Rectangle.create(0, 0, 0, 0)
            elif isinstance(region, Polygon):
                return trax.Polygon.create([(x.x, x.y) for x in region.points])
            elif isinstance(region, np.ndarray):
                return trax.Mask.create(region)
            else:
                return trax.Rectangle.create(region.x, region.y, region.width, region.height)

        if not _USE_TRAX:
            
            if not self._multiobject:
                status = [status]
            
            assert len(self._object_keys) == len(status), "Number of status entries must match the number of objects"
            for key, state in enumerate(status):
                self._object_trajectory[key].append(state)
            
        else:

            if not self._multiobject:
                status = [(convert_trax(status))]
            else:
                assert isinstance(status, (list, tuple))
                status = [(convert_trax(x), {}) for x in status]

            self._trax.status(status, {})

    def frame(self):
        """
        Get a frame (image path) from client

        Returns:
            absolute path of the image
        """
        
        if not _USE_TRAX:
            if self._position >= len(self._frames):
                return None
            frame = self._frames[self._position]
            self._position += 1
            return frame
        
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        # Only the first frame can declare new objects for now
        assert request.objects is None or len(request.objects) == 0

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None

    def quit(self):
        """ Quit the tracker"""
        if _USE_TRAX and hasattr(self, '_trax'):
            self._trax.quit()
        if not _USE_TRAX:
            for key, trajectory in zip(self._object_keys, self._object_trajectory):
                with open(f"output_{key}.txt", "w", encoding="utf-8") as f:
                    for state in trajectory:
                        f.write(_encode_region(state) + "\n")
            self._object_trajectory = []

    def __del__(self):
        """ Destructor for the tracker, calls quit. """
        self.quit()

class VOTManager(object):
    """ VOT Manager is provides a simple interface for running multiple single object trackers in parallel. Trackers should implement a factory interface. """

    def __init__(self, factory, region_format, channels=None):
        """ Constructor for the manager. 
        The factory should be a callable that accepts two arguments: image and region and returns a callable that accepts a single argument (image) and returns a region.

        Args:
            factory: Factory function for creating trackers
            region_format: Region format options
            channels: Channels that are supported by the tracker
        """
        self._handle = VOT(region_format, channels, multiobject=True)
        self._factory = factory

    def run(self):
        """ Run the tracker, the tracking loop is implemented in this function, so it will block until the client terminates the connection."""
        objects = self._handle.objects()

        # Process the first frame
        image = self._handle.frame()
        if not image:
            return

        trackers = [self._factory(image, object) for object in objects]

        while True:

            image = self._handle.frame()
            if not image:
                break

            status = [tracker(image) for tracker in trackers]

            self._handle.report(status)

        self._handle.quit()
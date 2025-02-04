import React, { useState, useEffect, useRef } from 'react';
import { getStroke } from 'perfect-freehand'; // Library for generating strokes based on input points.
import './tldraw.css'; // Importing the CSS file for styles.
import { getSvgPathFromStroke } from './utils'; // Utility function to generate SVG paths from stroke data.

const colors = ['#161a1d', '#ffb703', '#fb8500', '#023047', '#219ebc', '#d62828', '#9a031e', '#5f0f40', '#006400', '#8ac926', '#f28482'];
const sizes = [2, 4, 8, 16, 24, 32];

const TlDrawCanvasComponent = () => {
  const [lines, setLines] = useState([]);
  const [currentLine, setCurrentLine] = useState([]);
  const [currentColor, setCurrentColor] = useState(colors[0]);
  const [currentSize, setCurrentSize] = useState(sizes[2]);
  const [isEraser, setIsEraser] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isSilent, setIsSilent] = useState(false); // State to track if we are in a silent period

  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const lastPointRef = useRef([window.innerWidth / 2, window.innerHeight / 2]);
  const randomWalkIntervalRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = 'source-over';
    ctxRef.current = ctx;
  }, []);

  const generateRandomWalkPoint = (lastPoint, radius) => {
    const [lastX, lastY] = lastPoint;
    const angle = Math.random() * 2 * Math.PI;
    const distance = Math.random() * radius;
    const newX = lastX + distance * Math.cos(angle);
    const newY = lastY + distance * Math.sin(angle);
    return [newX, newY];
  };

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
  
    ws.onopen = () => {
      console.log('Connected to WebSocket server');
    };
  
    ws.onmessage = (event) => {
      const [x, y, newLineFlag] = event.data.split(',').map(Number);
  
      if (newLineFlag === 1) {
        setCurrentLine([]); // Start a new line
        setIsSilent(true); // Enter silent period, stop random walk
      } else {
        handleDrawing(x, y); // Continue drawing with the new coordinates
        lastPointRef.current = [x, y]; // Update the last drawn point
        setIsSilent(false); // Exit silent period, resume random walk
      }
    };
  
    return () => {
      console.log('Closing WebSocket connection');
      ws.close();
    };
  }, []);

  useEffect(() => {
    randomWalkIntervalRef.current = setInterval(() => {
      if (!isSilent) { // Only perform random walk if not in silent mode
        const [lastX, lastY] = lastPointRef.current;
        // const [newX, newY] = generateRandomWalkPoint([lastX, lastY], 13); // Adjust radius as needed
        const [newX, newY] = generateRandomWalkPoint([lastX, lastY], 0); // Adjust radius as needed
        handleDrawing(newX, newY); // Draw the random walk point
        lastPointRef.current = [newX, newY]; // Update the last point to the new random point
      }
    }, 50); // Adjust the interval timing as needed

    return () => {
      clearInterval(randomWalkIntervalRef.current); // Clear interval on component unmount
    };
  }, [isSilent]); // Rerun effect when isSilent changes

  const handleDrawing = (x, y) => {
    const newPoint = [x, y, Date.now()];
    setCurrentLine((prevLine) => [...prevLine, newPoint]);
  };

  useEffect(() => {
    if (currentLine.length > 1) {
      drawLine(); // Draw the line if there is more than one point.
    }
  }, [currentLine]);

  const drawLine = () => {
    const ctx = ctxRef.current;
    ctx.save();

    const stroke = getStroke(currentLine, {
      size: currentSize,
      thinning: 0.5,
      smoothing: 0.5,
      streamline: 0.5,
      easing: (t) => Math.sin((t * Math.PI) / 2),
      start: {
        taper: 20,
        easing: (t) => t * t,
        cap: true
      },
      end: {
        taper: 20,
        easing: (t) => t * t,
        cap: true
      }
    });

    const pathData = getSvgPathFromStroke(stroke);
    const path = new Path2D(pathData);

    ctx.globalCompositeOperation = 'source-atop';
    ctx.fillStyle = currentColor;
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    ctx.restore();
    ctx.strokeStyle = currentColor;
    ctx.lineWidth = 1;
    ctx.stroke(path);
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    const { offsetX, offsetY } = e.nativeEvent;
    handleDrawing(offsetX, offsetY);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = e.nativeEvent;
    handleDrawing(offsetX, offsetY);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    setLines((prevLines) => [...prevLines, { points: currentLine, color: currentColor, size: currentSize }]);
    setCurrentLine([]);
  };

  return (
    <div className="tldraw-container">
      <div className="controls">
        <div className="button-group">
          {colors.map((color, index) => (
            <button
              key={index}
              className="color-button"
              style={{ backgroundColor: color }}
              onClick={() => {
                setCurrentColor(color);
                setIsEraser(false);
              }}
            />
          ))}
          <button
            className={`eraser-button ${isEraser ? 'active' : ''}`}
            onClick={() => setIsEraser(!isEraser)}
          >
            Eraser
          </button>
        </div>
        <div className="button-group">
          {sizes.map((size, index) => (
            <button
              key={index}
              className="size-button"
              style={{ backgroundColor: '#333' }}
              onClick={() => setCurrentSize(size)}
            >
              {size}
            </button>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseOut={handleMouseUp}
      />
    </div>
  );
};

export default TlDrawCanvasComponent;
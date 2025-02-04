import React, { useRef, useEffect } from 'react';

const GridCanvas = () => {
    const canvasRef = useRef(null);

    // Constants for the grid
    const numColumns = 50;
    const numRows = 21;

    const drawGrid = (ctx, width, height) => {
        const columnWidth = width / numColumns;
        const rowHeight = height / numRows;

        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 0.5;

        // Draw vertical lines
        for (let i = 0; i <= numColumns; i++) {
            ctx.beginPath();
            ctx.moveTo(i * columnWidth, 0);
            ctx.lineTo(i * columnWidth, height);
            ctx.stroke();
        }

        // Draw horizontal lines
        for (let j = 0; j <= numRows; j++) {
            ctx.beginPath();
            ctx.moveTo(0, j * rowHeight);
            ctx.lineTo(width, j * rowHeight);
            ctx.stroke();
        }
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const resizeCanvas = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            drawGrid(ctx, canvas.width, canvas.height);
        };

        // Initial draw and setup resize event
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        return () => {
            window.removeEventListener('resize', resizeCanvas);
        };
    }, []);

    return (
        <canvas ref={canvasRef} className="grid-canvas"/>
    );
};

export default GridCanvas;

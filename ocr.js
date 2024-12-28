import Lens from 'chrome-lens-ocr';
import fs from 'fs/promises';
import path from 'path';
import { promisify } from 'util';
import { exec } from 'child_process';
import cliProgress from 'cli-progress';
import { JSDOM } from 'jsdom';

const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
const execAsync = promisify(exec);
const lens = new Lens();

const imagesDirectory = path.join(process.cwd(), 'regionimages');
const [, , resultFilePath, chunkNum, chunkSize, outputDir] = process.argv;

function parseImageFilename(filename) {
    const parts = filename.split('_');
    const pageWithinChunk = parseInt(parts[0]);
    const actualPageNum = (parseInt(chunkNum) - 1) * parseInt(chunkSize) + pageWithinChunk + 1;
    const isTableCell = parts.includes('a');
    if (isTableCell) {
        return {
            pageNum: actualPageNum,
            orderNum: parseInt(parts[1]),
            label: 'a',
            tableNum: parseInt(parts[1]),
            row: parseInt(parts[3]),
            col: parseInt(parts[4].split('.')[0]),
            fullPath: path.join(imagesDirectory, filename),
            uniqueTableId: `${actualPageNum}_${parts[1]}`
        };
    } else {
        return {
            pageNum: actualPageNum,
            orderNum: parseInt(parts[1]),
            label: parts[2].split('.')[0],
            fullPath: path.join(imagesDirectory, filename)
        };
    }
}

async function processImageWithRetry(fullPath, file, maxRetries = 25) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            const result = await lens.scanByFile(fullPath);
            if (result && result.segments) {
                let ocrText = result.segments.map(segment => segment.text).join(' ');
                ocrText = Buffer.from(ocrText).toString('utf8');
                return { file, ocrText, attempts: attempt };
            } else {
                throw new Error('Invalid OCR result format.');
            }
        } catch (error) {
            if (attempt === maxRetries) {
                return { file, ocrText: '[OCR_FAILED]', attempts: attempt };
            }
        }
    }
}

const tableStructures = {};

function manageTableStructure(cellInfo) {
    const { uniqueTableId, row, col, ocrText } = cellInfo;
    if (!tableStructures[uniqueTableId]) {
        tableStructures[uniqueTableId] = { cells: {}, maxRow: 0, maxCol: 0 };
    }
    const table = tableStructures[uniqueTableId];
    table.cells[`${row}_${col}`] = ocrText;
    table.maxRow = Math.max(table.maxRow, row);
    table.maxCol = Math.max(table.maxCol, col);
}

async function processImagesWithQueue(images, maxConcurrentJobs, ocrResults) {
    const queue = images.filter(img => !['f', 'p'].includes(img.label));
    let jobsInProgress = 0;
    let processedCount = 0;
    
    progressBar.start(queue.length, 0);

    const processNextImage = () => {
        if (queue.length === 0) {
            if (jobsInProgress === 0) return Promise.resolve();
            return;
        }

        const image = queue.shift();
        jobsInProgress++;

        return processImageWithRetry(image.fullPath, image.fullPath)
            .then((result) => {
                if (image.label === 'a') {
                    manageTableStructure({
                        uniqueTableId: image.uniqueTableId,
                        row: image.row,
                        col: image.col,
                        ocrText: result.ocrText
                    });
                } else {
                    ocrResults[`${image.pageNum}_${image.orderNum}_${image.label}`] = {
                        ...image,
                        ocrText: result.ocrText
                    };
                }
                processedCount++;
                progressBar.update(processedCount);
            })
            .catch((error) => {
                throw error;
            })
            .finally(() => {
                jobsInProgress--;
                return processNextImage();
            });
    };

    const promises = [];
    for (let i = 0; i < maxConcurrentJobs; i++) {
        promises.push(processNextImage());
    }

    await Promise.all(promises);
}

function createHtmlElement(label, content, dom) {
    const document = dom.window.document;
    let element;
    
    switch (label) {
        case 'c':
            element = document.createElement('figcaption');
            element.className = 'caption';
            element.textContent = content;
            break;
        case 'fo':
            element = document.createElement('div');
            element.className = 'footnote';
            element.innerHTML = content;
            break;
        case 'fr':
            element = document.createElement('div');
            element.className = 'formula';
            element.innerHTML = content;
            break;
        case 'l':
            element = document.createElement('li');
            element.className = 'list-item';
            element.textContent = content;
            break;
        case 'pf':
            element = document.createElement('footer');
            element.className = 'page-footer';
            element.textContent = content;
            break;
        case 'ph':
            element = document.createElement('header');
            element.className = 'page-header';
            element.textContent = content;
            break;
        case 'p':
        case 'f':
            element = document.createElement('figure');
            element.className = label === 'p' ? 'picture' : 'figure';
            const img = document.createElement('img');
            img.src = content;
            element.appendChild(img);
            break;
        case 's':
            element = document.createElement('h2');
            element.className = 'section-header';
            element.textContent = content;
            break;
        case 'a':
            element = generateCompleteTableHtml(content, dom);
            break;
        case 'e':
            element = document.createElement('p');
            element.className = 'text';
            element.textContent = content;
            break;
        case 'i':
            element = document.createElement('h1');
            element.className = 'title';
            element.textContent = content;
            break;
        default:
            element = document.createElement('div');
            element.textContent = content;
    }
    
    return element;
}

async function embedImageInHtml(imagePath) {
    const imageData = await fs.readFile(imagePath);
    const base64Image = imageData.toString('base64');
    const mimeType = path.extname(imagePath).slice(1);
    return `data:image/${mimeType};base64,${base64Image}`;
}

function generateCompleteTableHtml(tableId, dom) {
    const table = tableStructures[tableId];
    if (!table) {
        return null;
    }
    
    const document = dom.window.document;
    const tableElement = document.createElement('table');
    tableElement.className = 'data-table';

    for (let r = 0; r <= table.maxRow; r++) {
        const row = document.createElement('tr');
        for (let c = 0; c <= table.maxCol; c++) {
            const cell = document.createElement('td');
            cell.textContent = table.cells[`${r}_${c}`] || '';
            row.appendChild(cell);
        }
        tableElement.appendChild(row);
    }

    return tableElement;
}

async function processImages(directory, outputDir) {
  try {
      const files = await fs.readdir(directory, { encoding: 'utf8' });
      const pngFiles = files.filter(file => file.endsWith('.png'));
      const parsedImages = pngFiles.map(parseImageFilename).sort((a, b) => {
          if (a.pageNum !== b.pageNum) return a.pageNum - b.pageNum;
          return a.orderNum - b.orderNum;
      });

      const ocrResults = {};
      await processImagesWithQueue(parsedImages, 50, ocrResults);

      progressBar.stop();

      // Process results by page
      const pageContents = {};
      const dom = new JSDOM('<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body></body></html>');

      for (const image of parsedImages) {
          const pageNum = image.pageNum;
          if (!pageContents[pageNum]) {
              pageContents[pageNum] = [];
          }

          let element = null;
          
          if (['f', 'p'].includes(image.label)) {
              // Skip adding images entirely instead of converting to base64
              continue;
          } else if (image.label === 'a') {
              if (!pageContents[pageNum].some(content => 
                  content.tableId === image.uniqueTableId)) {
                  const tableElement = generateCompleteTableHtml(image.uniqueTableId, dom);
                  if (tableElement) {
                      pageContents[pageNum].push({
                          orderNum: image.orderNum,
                          element: tableElement,
                          tableId: image.uniqueTableId
                      });
                  }
              }
          } else {
              const result = ocrResults[`${image.pageNum}_${image.orderNum}_${image.label}`];
              if (result) {
                  element = createHtmlElement(image.label, result.ocrText, dom);
                  pageContents[pageNum].push({
                      orderNum: image.orderNum,
                      element: element
                  });
              }
          }
      }

      // Write individual HTML files for each page
      const sortedPageNumbers = Object.keys(pageContents).sort((a, b) => parseInt(a) - parseInt(b));
      
      for (const pageNum of sortedPageNumbers) {
          const pageContainer = dom.window.document.createElement('div');
          pageContainer.className = 'page';

          // Sort elements by order number and append to page
          pageContents[pageNum]
              .sort((a, b) => a.orderNum - b.orderNum)
              .forEach(content => {
                  if (content.element) {
                      // Remove any remaining base64 images
                      const element = content.element.cloneNode(true);
                      const imgs = element.getElementsByTagName && element.getElementsByTagName('img');
                      if (imgs) {
                          for (let i = imgs.length - 1; i >= 0; i--) {
                              const img = imgs[i];
                              if (img.src && img.src.startsWith('data:image/')) {
                                  img.parentNode.removeChild(img);
                              }
                          }
                      }
                      pageContainer.appendChild(element);
                  }
              });

          // Clean up empty figure elements
          const figures = pageContainer.getElementsByTagName('figure');
          if (figures) {
              for (let i = figures.length - 1; i >= 0; i--) {
                  const figure = figures[i];
                  if (!figure.hasChildNodes()) {
                      figure.parentNode.removeChild(figure);
                  }
              }
          }

          // Write to file
          const outputPath = path.join(outputDir, `${pageNum}.html`);
          await fs.writeFile(outputPath, pageContainer.outerHTML, 'utf8');
      }

  } catch (error) {
      progressBar.stop();
      console.error('Error in processImages:', error);
      throw error;
  }
}

if (!outputDir) {
    console.error('Invalid command-line arguments. Usage: node ocr.js <resultFilePath> <chunkNum> <chunkSize> <outputDir>');
    process.exit(1);
}

fs.mkdir(outputDir, { recursive: true })
    .then(() => processImages(imagesDirectory, outputDir))
    .then(() => {
        console.log('Image processing and HTML generation complete.');
    })
    .catch(error => {
        console.error('Error:', error);
        process.exit(1);
    });
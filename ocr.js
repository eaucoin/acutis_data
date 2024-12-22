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
const [, , resultFilePath, chunkNum, chunkSize] = process.argv;

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

function createHtmlElement(label, content) {
  const dom = new JSDOM('', { contentType: 'text/html; charset=utf-8' });
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
      element = generateCompleteTableHtml(content);
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

function generateCompleteTableHtml(tableId) {
  const table = tableStructures[tableId];
  if (!table) {
    return null;
  }
  
  const dom = new JSDOM();
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

function insertTableInOrderedContent(pageContent, tableHtml, tableOrderNum) {
  let insertIndex = pageContent.findIndex(item => item.orderNum > tableOrderNum);
  if (insertIndex === -1) insertIndex = pageContent.length;
  pageContent.splice(insertIndex, 0, { orderNum: tableOrderNum, html: tableHtml });
  return pageContent;
}

function createPageNumberElement(pageNum) {
  const dom = new JSDOM('', { contentType: 'text/html; charset=utf-8' });
  const document = dom.window.document;
  const element = document.createElement('div');
  element.className = 'page-number';
  element.textContent = `Page ${pageNum}`;
  return element;
}

async function generateHtmlDocument(processedResults) {
  const dom = new JSDOM('<!DOCTYPE html><html><head><meta charset="UTF-8"><style>.page-number { font-weight: bold; margin-bottom: 10px; }</style></head><body></body></html>', {
    contentType: 'text/html; charset=utf-8'
  });
  const { document } = dom.window;

  const pageContents = {};
  const processedTables = new Set();

  for (const result of processedResults) {
    if (!pageContents[result.pageNum]) {
      pageContents[result.pageNum] = [];
    }

    let element;
    if (['f', 'p'].includes(result.label)) {
      const imageData = await embedImageInHtml(result.fullPath);
      element = createHtmlElement(result.label, imageData);
    } else if (result.label === 'a' && !processedTables.has(result.uniqueTableId)) {
      const tableHtml = generateCompleteTableHtml(result.uniqueTableId);
      if (tableHtml) {
        pageContents[result.pageNum] = insertTableInOrderedContent(pageContents[result.pageNum], tableHtml, result.orderNum);
        processedTables.add(result.uniqueTableId);
      }
      continue;
    } else if (result.label !== 'a') {
      element = createHtmlElement(result.label, result.ocrText);
    } else {
      continue;
    }

    if (element) {
      pageContents[result.pageNum].push({ orderNum: result.orderNum, html: element });
    }
  }

  const sortedPageNumbers = Object.keys(pageContents).sort((a, b) => parseInt(a) - parseInt(b));

  for (const pageNum of sortedPageNumbers) {
    const pageContainer = document.createElement('div');
    pageContainer.className = 'page';

    const pageNumberElement = createPageNumberElement(pageNum);
    pageContainer.appendChild(pageNumberElement);

    pageContents[pageNum].sort((a, b) => a.orderNum - b.orderNum);
    for (const item of pageContents[pageNum]) {
      pageContainer.appendChild(item.html);
    }

    document.body.appendChild(pageContainer);
  }

  return dom.serialize();
}

async function writeHtmlFile(html, outputPath) {
  await fs.writeFile(outputPath, html, 'utf8');
}

async function processImages(directory) {
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

    const processedResults = parsedImages.map(img => {
      if (['f', 'p'].includes(img.label)) {
        return img;
      } else if (img.label === 'a') {
        return img;
      } else {
        return ocrResults[`${img.pageNum}_${img.orderNum}_${img.label}`] || img;
      }
    });

    const html = await generateHtmlDocument(processedResults);
    await writeHtmlFile(html, resultFilePath);

  } catch (error) {
    progressBar.stop();
    console.error('Error in processImages:', error);
  }
}

if (resultFilePath && chunkNum && chunkSize) {
  processImages(imagesDirectory)
    .then(() => console.log('Image processing and HTML generation complete.'))
    .catch(console.error);
} else {
  console.error('Invalid command-line arguments. Usage: node ocr.js <resultFilePath> <chunkNum> <chunkSize>');
}

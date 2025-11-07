import React, { useState, useCallback, useMemo } from 'react';

// NOTE: This URL should be updated to point to your running Flask backend.
const API_BASE_URL = 'http://localhost:5000';

const getStatusIcon = (currentStatus) => {
    switch (currentStatus) {
      case 'indexed':
      case 'success':
        return <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>; // Checkmark Circle
      case 'querying':
      case 'uploading':
        return <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 0012 4.004zm-14.28 11.23a8.001 8.001 0 0011.28 11.28l.582-.582m-15.356-11.23a8.001 8.001 0 00-11.28-11.28l-.582.582" /></svg>; // Spinner
      case 'error':
        return <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>; // X Circle
      default:
        return <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>; // Info Circle
    }
};

// --- Custom Component for Typing Animation ---
const TypingEffect = ({ content, speed = 25 }) => {
    const [displayedContent, setDisplayedContent] = useState('');
    const [currentIndex, setCurrentIndex] = useState(0);
    
    // Reset effect when content changes
    React.useEffect(() => {
        setDisplayedContent('');
        setCurrentIndex(0);
    }, [content]);

    // Typing logic
    React.useEffect(() => {
        if (!content || currentIndex >= content.length) return;

        const char = content[currentIndex];
        
        // Use a slight delay randomization to make it look more natural
        const delay = char === ' ' ? speed * 0.5 : speed * (0.8 + Math.random() * 0.4); 

        const timer = setTimeout(() => {
            setDisplayedContent(prev => prev + char);
            setCurrentIndex(prev => prev + 1);
        }, delay);

        return () => clearTimeout(timer);
    }, [content, currentIndex, speed]);

    // Pass the currently displayed content to the Markdown renderer
    return <BasicMarkdownRenderer content={displayedContent} />;
};
// ---------------------------------------------------

// --- Enhanced Custom Component for Markdown Rendering ---
// This version supports basic headings (H1-H3), unordered lists, bold, and italic.
const BasicMarkdownRenderer = ({ content }) => {
    if (!content) return null;

    // A. Inline Rendering Helper (handles bold, italic, soft breaks)
    const renderInline = (text) => {
        let parts = [text];

        // Process **bold**
        parts = parts.flatMap(part => {
            if (typeof part !== 'string') return part;
            const segments = part.split(/(\*\*(.*?)\*\*)/g).filter(s => s.length > 0);
            return segments.map((segment, i) => {
                const match = segment.match(/\*\*(.*?)\*\*/);
                if (match) { return <strong key={`b-${i}`}>{match[1]}</strong>; }
                return segment;
            });
        });

        // Process *italic*
        parts = parts.flatMap(part => {
            if (typeof part !== 'string') return part;
            const segments = part.split(/(\*(.*?)\*)/g).filter(s => s.length > 0);
            return segments.map((segment, i) => {
                const match = segment.match(/\*(.*?)\*/);
                if (match) { return <em key={`i-${i}`}>{match[1]}</em>; }
                return segment;
            });
        });

        // Handle single newlines (soft breaks)
        parts = parts.flatMap(part => {
            if (typeof part !== 'string') return part;
            const segments = part.split('\n');
            return segments.flatMap((segment, i) => [
                segment,
                i < segments.length - 1 ? <br key={`br-${i}`} /> : null
            ]).filter(Boolean);
        });

        return parts;
    };


    // B. Block-level parsing
    const lines = content.split('\n');
    let blocks = [];
    let currentList = null;

    lines.forEach((line, index) => {
        const trimmedLine = line.trim();
        const key = `block-${index}`;

        if (trimmedLine.match(/^#{1,3}\s/)) {
            // Heading detection (H1, H2, H3)
            if (currentList) {
                blocks.push(<ul key={`ul-${blocks.length}`} className="space-y-1 mb-4 ml-6 list-disc text-gray-700">{currentList}</ul>);
                currentList = null;
            }
            const match = trimmedLine.match(/^(#+)\s(.*)/);
            if (match) {
                const level = match[1].length;
                const text = match[2];
                if (level === 1) blocks.push(<h1 key={key} className="text-3xl font-extrabold text-gray-800 pt-6 mt-4 mb-2">{text}</h1>);
                else if (level === 2) blocks.push(<h2 key={key} className="text-2xl font-bold text-gray-700 pt-4 mt-3 mb-2">{text}</h2>);
                else if (level === 3) blocks.push(<h3 key={key} className="text-xl font-semibold text-gray-700 pt-3 mt-2 mb-1">{text}</h3>);
            }
        } else if (trimmedLine.match(/^[*-]\s/)) {
            // Unordered List detection
            const text = trimmedLine.replace(/^[*-]\s/, '');
            const listItem = <li key={`li-${index}`} className="pl-2">{renderInline(text)}</li>;
            if (!currentList) {
                currentList = [listItem];
            } else {
                currentList.push(listItem);
            }
        } else if (trimmedLine === '' && currentList) {
            // End of list block
            blocks.push(<ul key={`ul-${blocks.length}`} className="space-y-1 mb-4 ml-6 list-disc text-gray-700">{currentList}</ul>);
            currentList = null;
        } else if (trimmedLine !== '') {
            // Paragraph
            if (currentList) {
                blocks.push(<ul key={`ul-${blocks.length}`} className="space-y-1 mb-4 ml-6 list-disc text-gray-700">{currentList}</ul>);
                currentList = null;
            }
            blocks.push(<p key={key} className="text-gray-900 leading-relaxed text-lg font-medium pt-1">{renderInline(line)}</p>);
        }
    });

    // Commit any open list block at the end of content
    if (currentList) {
        blocks.push(<ul key={`ul-${blocks.length}`} className="space-y-1 mb-4 ml-6 list-disc text-gray-700">{currentList}</ul>);
    }

    // Set scrollable container and apply default spacing
    return (
        <div className="space-y-3 max-h-[500px] overflow-y-auto">
            {blocks}
        </div>
    );
};
// ---------------------------------------------------


const StepVisualization = ({ trace }) => {
    if (!trace || !trace.steps) return null;

    const getColor = (status) => {
        switch (status) {
            case 'Completed': return 'bg-emerald-500';
            case 'processing': return 'bg-yellow-500';
            case 'error': return 'bg-red-500';
            default: return 'bg-gray-400';
        }
    };

    return (
        <div className="space-y-4 p-5 border border-gray-200 rounded-xl bg-white shadow-lg">
            <h3 className="text-xl font-bold text-gray-800 border-b pb-2 mb-3">Process Trace ({trace.elapsed_time?.overall})</h3>
            
            <div className="space-y-3">
                {trace.steps.map((step, index) => (
                    <div key={index} className="flex items-center space-x-4">
                        <div className={`w-3 h-3 rounded-full flex-shrink-0 ${getColor(step.status)}`}></div>
                        <div className="flex-grow">
                            <p className="text-sm font-medium text-gray-700">{step.name}</p>
                        </div>
                        <p className="text-xs text-gray-500 font-mono flex-shrink-0">{step.duration}</p>
                    </div>
                ))}
            </div>

            <h4 className="text-lg font-semibold mt-6 text-gray-700 pt-3 border-t">Source Context Snippets</h4>
            <pre className="text-xs p-4 bg-gray-50 border border-gray-100 rounded-lg whitespace-pre-wrap max-h-56 overflow-y-auto font-mono text-gray-600">
                {trace.source_context || "Context not available. Check steps for failure details."}
            </pre>
        </div>
    );
};

const App = () => {
    
    const [status, setStatus] = useState('idle'); 
    const [message, setMessage] = useState('Welcome! Please upload a PDF document to begin indexing.');
    const [file, setFile] = useState(null);
    const [documentName, setDocumentName] = useState('');
    const [question, setQuestion] = useState('');
    const [results, setResults] = useState(null);
    

    const isProcessing = status.includes('uploading') || status.includes('querying');

    const uploadFile = useCallback(async () => {
        if (!file) {
            setMessage("Please select a file first.");
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);

        setStatus('uploading');
        setMessage(`Uploading ${file.name}...`);
        setDocumentName('');
        setResults(null);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                setStatus('indexed');
                setMessage(`Success! Document indexed into ${data.chunks_processed} chunks in ${data.indexing_time}. Ask a question below.`);
                setDocumentName(data.document_name);
            } else {
                setStatus('error');
                setMessage(`Indexing failed: ${data.message || 'Unknown error.'}`);
            }
        } catch (error) {
            setStatus('error');
            setMessage(`Network Error: Could not connect to backend. Is the Flask server running on ${API_BASE_URL}?`);
            console.error("Upload Error:", error);
        }
    }, [file]);

    const queryDocument = useCallback(async () => {
        if (!question.trim()) {
            setMessage("Please enter a question.");
            return;
        }
        
        setStatus('querying');
        setMessage(`Searching document for: "${question}"...`);
        setResults(null);

        try {
            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                setStatus('success');
                setMessage(`Query complete. Answer generated by the RAG system.`);
                setResults(data);
            } else {
                setStatus('error');
                setMessage(`Query failed: ${data.message || data.error || 'Unknown error.'}`);
                setResults(data);
            }
        } catch (error) {
            setStatus('error');
            setMessage(`Network Error during query. Check console for details.`);
            console.error("Query Error:", error);
        }
    }, [question]);

    const statusColor = useMemo(() => {
        switch (status) {
            case 'indexed':
            case 'success':
                return 'bg-emerald-50 text-emerald-700 border-emerald-300';
            case 'querying':
            case 'uploading':
                return 'bg-blue-50 text-blue-700 border-blue-300';
            case 'error':
                return 'bg-red-50 text-red-700 border-red-300';
            default:
                return 'bg-gray-100 text-gray-600 border-gray-300';
        }
    }, [status]);

    return (
        <div className="min-h-screen bg-gray-50 p-4 sm:p-10 font-sans">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
                    <span className="text-blue-600">Semantic Contextual PDF Retrieval</span> (SCPR)
                </h1>
                <p className="text-lg text-gray-500 mb-8">Private, Fact-Checked Q&A for your Documents.</p>

                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Control Panel */}
                    <div className="lg:col-span-1 space-y-6">
                        
                        {/* Status Message */}
                        <div className={`p-4 rounded-xl border-2 ${statusColor} shadow-lg transition-all flex items-start space-x-3`}>
                            <div className="pt-1 flex-shrink-0">
                                {getStatusIcon(status)}
                            </div>
                            <div>
                                <p className="text-sm font-extrabold">Status: {status.toUpperCase()}</p>
                                <p className="text-xs mt-1">{message}</p>
                                {documentName && (
                                    <p className="text-xs mt-2 font-mono break-all opacity-75">Doc: {documentName}</p>
                                )}
                            </div>
                        </div>

                        {/* Step 1: Upload */}
                        <div className="p-6 bg-white rounded-xl shadow-xl border border-gray-100">
                            <h2 className="text-xl font-semibold mb-4 text-gray-800">1. Index Document (PDF)</h2>
                            <input
                                type="file"
                                accept=".pdf"
                                onChange={(e) => setFile(e.target.files[0])}
                                className="w-full text-sm text-gray-500
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-full file:border-0
                                file:text-sm file:font-bold
                                file:bg-blue-50 file:text-blue-700
                                hover:file:bg-blue-100 transition-colors focus:outline-none"
                            />
                            <button
                                onClick={uploadFile}
                                disabled={!file || isProcessing}
                                className="mt-4 w-full px-4 py-2 bg-blue-600 text-white font-bold rounded-lg shadow-lg hover:bg-blue-700 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {status === 'uploading' ? 'Indexing...' : 'Upload & Index'}
                            </button>
                        </div>

                        {/* Step 2: Query */}
                        <div className="p-6 bg-white rounded-xl shadow-xl border border-gray-100">
                            <h2 className="text-xl font-semibold mb-4 text-gray-800">2. Ask a Question</h2>
                            <textarea
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                placeholder="E.g., What is the capital expenditure limit for Q4?"
                                rows="3"
                                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-purple-500 focus:border-purple-500 resize-none transition-shadow focus:shadow-md"
                                disabled={status !== 'indexed' && status !== 'success'}
                            />
                            <button
                                onClick={queryDocument}
                                disabled={status !== 'indexed' && status !== 'success' || isProcessing}
                                className="mt-4 w-full px-4 py-2 bg-purple-600 text-white font-bold rounded-lg shadow-lg hover:bg-purple-700 transition duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {status === 'querying' ? 'Generating Answer...' : 'Generate Answer'}
                            </button>
                        </div>
                    </div>

                    {/* Results Panel */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Glowing Gradient Card for Answer */}
                        <div className="p-6 bg-white rounded-xl shadow-2xl border border-gray-100 min-h-[180px]
                            relative group overflow-hidden">
                            
                            {/* Glowing Border effect setup - sits behind content */}
                            <div className="absolute -inset-1 rounded-xl pointer-events-none z-0">
                                <div className="absolute inset-0.5 rounded-xl bg-white"></div> {/* Inner white mask */}
                                {/* The actual gradient ring, blurred and animating */}
                                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-purple-500 via-blue-500 to-emerald-500 
                                    opacity-0 group-hover:opacity-70 transition-opacity duration-500 blur-lg animate-pulse"></div>
                            </div>
                            
                            {/* Content wrapper to ensure content is above the glow and visible */}
                            <div className="relative z-10"> 
                                <h2 className="text-2xl font-bold text-gray-800 mb-4 border-b pb-2">Generated Answer</h2>
                                
                                {isProcessing && (
                                    <div className="flex items-center justify-center p-8 text-blue-600">
                                        <svg className="animate-spin -ml-1 mr-3 h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <p className="font-semibold text-lg">Processing query and generating response...</p>
                                    </div>
                                )}
                                
                                {results && results.final_answer && (
                                    <div className="space-y-4">
                                        {/* Use TypingEffect to display the answer with animation and markdown */}
                                        <TypingEffect content={results.final_answer} />

                                        {/* Grounding Status */}
                                        <div className="flex items-center space-x-2 p-3 bg-emerald-50 rounded-lg border border-emerald-300 text-emerald-800 mt-6">
                                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                            <p className="text-sm font-semibold">Answer successfully grounded in document context.</p>
                                        </div>
                                    </div>
                                )}
                                {results && results.status === 'error' && (
                                    <p className="text-red-600 text-sm p-4 bg-red-50 rounded-lg border border-red-300 font-semibold">{results.message}</p>
                                )}
                            </div>
                        </div>

                        {/* Process Trace */}
                        {results && <StepVisualization trace={results} />}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;


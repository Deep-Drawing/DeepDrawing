import React, { createContext, useContext, useState } from 'react';

// Create the Context
const EditorContext = createContext();

// Export the custom hook for using the context
export const useEditor = () => useContext(EditorContext);

// Context provider component
export const EditorProvider = ({ children }) => {
  const [editor, setEditor] = useState(null);

  return (
    <EditorContext.Provider value={{ editor, setEditor }}>
      {children}
    </EditorContext.Provider>
  );
};

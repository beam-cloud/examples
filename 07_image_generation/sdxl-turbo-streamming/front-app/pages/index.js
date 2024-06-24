import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Home() {
  const [text, setText] = useState('');
  const [imageUrl, setImageUrl] = useState('');

  useEffect(() => {
    const fetchResponse = async () => {
      if (text) {
        try {
          const response = await axios.post(
            'http://localhost:8000/sdxl_streaming',
            { content: text },
            { responseType: 'blob' } // Ensure the response is handled as a Blob
          );
          
          // Create a URL for the binary data
          const imageBlob = response.data;
          const imageObjectUrl = URL.createObjectURL(imageBlob);
          setImageUrl(imageObjectUrl);
        } catch (error) {
          console.error('Error fetching response:', error);
        }
      }
    };
    fetchResponse();
  }, [text]);

  const handleChange = (e) => {
    setText(e.target.value);
  };

  return (
    <div>
      <h1>Stable Diffusion Stream</h1>
      <input type="text" value={text} onChange={handleChange} />
      {imageUrl && <img src={imageUrl} alt="Generated" />}
    </div>
  );
}

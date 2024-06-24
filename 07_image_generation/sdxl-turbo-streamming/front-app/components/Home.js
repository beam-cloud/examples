// Home.js
import { useState, useEffect } from 'react';
import axios from 'axios';
import styles from '../styles/Home.module.css';

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
            { responseType: 'blob' }
          );
          
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
    <div className={styles.container}>
      <h1 className={styles.title}>Stable Diffusion Stream</h1>
      <input className={styles.inputField}  type="text" value={text} onChange={handleChange} />
      {imageUrl && <img className={styles.image} src={imageUrl} alt="Generated" />}
    </div>
  );
}

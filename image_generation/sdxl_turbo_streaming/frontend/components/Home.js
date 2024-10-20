import { useState, useEffect } from 'react';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [text, setText] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [debouncedText, setDebouncedText] = useState('');

  useEffect(() => {
    const delay = 500; // Delay in milliseconds
    const timeoutId = setTimeout(() => {
      setDebouncedText(text);
    }, delay);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [text]);

  useEffect(() => {
    const fetchResponse = async () => {
      if (debouncedText) {
        try {
          const response = await fetch(
            process.env.NEXT_PUBLIC_BEAM_API_URL,
            {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${process.env.NEXT_PUBLIC_BEAM_AUTH_TOKEN}`,
                'Connection': 'keep-alive',
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({ prompt: debouncedText })
            }
          );

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          console.log(data)
          // const imageObjectUrl = URL.createObjectURL(blob);
          setImageUrl(data.image);
        } catch (error) {
          console.error('Error fetching response:', error);
        }
      }
    };

    fetchResponse();
  }, [debouncedText]);

  const handleChange = (e) => {
    setText(e.target.value);
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Stable Diffusion Stream</h1>
      <input className={styles.inputField} type="text" value={text} onChange={handleChange} />
      {imageUrl && <img className={styles.image} src={imageUrl} alt="Generated" />}
    </div>
  );
}

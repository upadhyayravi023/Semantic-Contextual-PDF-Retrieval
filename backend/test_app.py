import pytest
import os
import shutil
import io
import gc
import time
import app

@pytest.fixture(scope="session")
def client():
    TEST_DB_PATH = 'chroma_db_test'
    TEST_UPLOAD_PATH = 'uploads_test'

    app.app.config['TESTING'] = True
    app.app.config['DB_PATH'] = TEST_DB_PATH
    app.app.config['UPLOAD_FOLDER'] = TEST_UPLOAD_PATH
    
    def cleanup_test_files():
        if app.vectorstore and hasattr(app.vectorstore, '_client'):
            try:
                app.vectorstore._client.reset()
            except:
                pass
        
        app.vectorstore = None
        app.embedding_function = None
        gc.collect()
        time.sleep(0.1)

        for i in range(3):
            try:
                if os.path.exists(TEST_DB_PATH):
                    shutil.rmtree(TEST_DB_PATH)
                if os.path.exists(TEST_UPLOAD_PATH):
                    shutil.rmtree(TEST_UPLOAD_PATH)
                break 
            except PermissionError:
                print(f"\n[Cleanup Retry {i+1}/3] PermissionError. Waiting 0.5s for file lock to release...")
                time.sleep(0.5)
            except Exception as e:
                print(f"\n[Cleanup Error] {e}")
                break 
        
        os.makedirs(TEST_UPLOAD_PATH, exist_ok=True)

    cleanup_test_files()

    with app.app.test_client() as client:
        yield client

    cleanup_test_files()


def test_upload_no_file(client):
    response = client.post('/upload', data={})
    json_data = response.get_json()
    
    assert response.status_code == 400
    assert "No file part" in json_data['message']


def test_upload_wrong_file_type(client):
    data = {
        'file': (io.BytesIO(b"this is a text file"), 'not_a_pdf.txt')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    json_data = response.get_json()
    
    assert response.status_code == 400
    assert "Invalid file type" in json_data['message']


def test_query_before_upload(client):
    response = client.post('/query', json={'question': 'test'})
    json_data = response.get_json()
    
    assert response.status_code == 400
    assert "Vector store is empty" in json_data['message']


def test_full_workflow_happy_path(client, mocker):
    mock_text = "The Eiffel Tower was built in 1889. It is located in Paris."
    mocker.patch('app.extract_text_from_pdf', return_value=mock_text)
    mocker.patch('app.GoogleGenerativeAIEmbeddings', return_value="fake-embedding-function")
    
    mock_llm_response = mocker.MagicMock()
    mock_llm_response.content = "The answer is 1889."
    
    mock_llm_class = mocker.patch('app.ChatGoogleGenerativeAI')
    mock_llm_class.return_value.invoke.return_value = mock_llm_response

    data = {
        'file': (io.BytesIO(b"this is a fake pdf"), 'eiffel_tower.pdf')
    }
    upload_response = client.post('/upload', data=data, content_type='multipart/form-data')
    upload_json = upload_response.get_json()
    
    assert upload_response.status_code == 200
    assert upload_json['status'] == 'success'
    assert upload_json['file_name'] == 'eiffel_tower.pdf'
    assert upload_json['total_chunks'] > 0

    query_response = client.post('/query', json={'question': 'When was the Eiffel Tower built?'})
    query_json = query_response.get_json()
    
    assert query_response.status_code == 200
    assert query_json['status'] == 'success'
    assert "1889" in query_json['final_answer']
    assert "Eiffel Tower" in query_json['source_context']
    
    mock_llm_class.return_value.invoke.assert_called_once()
    prompt_arg = mock_llm_class.return_value.invoke.call_args[0][0]
    assert "Eiffel Tower" in prompt_arg
    assert "When was the Eiffel Tower built?" in prompt_arg
